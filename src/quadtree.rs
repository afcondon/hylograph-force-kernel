//! Point quadtree for the many-body and collide forces.
//!
//! A port of the parts of d3-quadtree 3.0.1 that d3-force uses: `addAll`
//! (with `cover`), `visit` and `visitAfter`. The shape of the tree decides
//! which cells Barnes-Hut approximates and in which order points are met, so
//! it is reproduced exactly rather than approximately:
//!
//! - the root square has an integer origin and a power-of-two side, found by
//!   flooring the first corner of the extent and doubling until the second
//!   corner is covered (`cover.js`);
//! - a point goes right/down when `x >= xm` / `y >= ym`;
//! - points with identical coordinates share one leaf as a chain, newest
//!   first (`add.js`);
//! - `visit` is pre-order, children in the order 0, 1, 2, 3.
//!
//! Storage is an arena (indices, not boxes) so a tree can be rebuilt every
//! tick without allocating.

use crate::Real;

/// "No node here". Note that this value also has `LEAF_BIT` set, so test for
/// `NONE` before asking `is_leaf`.
pub const NONE: u32 = u32::MAX;

const LEAF_BIT: u32 = 1 << 31;

/// Splitting stops here even if two points are still in the same cell. Two
/// distinct float values are always separated long before this (the cell side
/// halves each level), so the cap only guards against a hang.
const MAX_SPLIT_DEPTH: u32 = 512;

/// Is this reference a leaf (a point, heading a chain of coincident points)?
#[inline]
pub fn is_leaf(q: u32) -> bool {
    q & LEAF_BIT != 0
}

/// The point index a leaf reference names (the head of its chain).
#[inline]
pub fn leaf_point(q: u32) -> usize {
    (q & !LEAF_BIT) as usize
}

#[inline]
fn leaf(point: usize) -> u32 {
    point as u32 | LEAF_BIT
}

/// A node together with the square it covers, as handed to `visit` callbacks.
#[derive(Clone, Copy)]
pub struct Quad {
    pub node: u32,
    pub x0: Real,
    pub y0: Real,
    pub x1: Real,
    pub y1: Real,
}

pub struct Quadtree {
    /// Root square. NaN while the tree is empty.
    pub x0: Real,
    pub y0: Real,
    pub x1: Real,
    pub y1: Real,
    /// `NONE`, a leaf reference, or the index of an internal node.
    pub root: u32,
    /// Internal nodes: four child references each.
    pub children: Vec<[u32; 4]>,
    /// Per point: the next point at identical coordinates, or `NONE`.
    pub next: Vec<u32>,
    /// Per point: the coordinates it was inserted at.
    pub xs: Vec<Real>,
    pub ys: Vec<Real>,
    /// Per internal node: payload written by the force that owns the tree
    /// (many-body: centre of charge and total charge; collide: `iv` is the
    /// largest radius below).
    pub ix: Vec<Real>,
    pub iy: Vec<Real>,
    pub iv: Vec<Real>,
    /// Per point: payload for the leaf this point heads.
    pub lv: Vec<Real>,
    /// Internal nodes with every parent before its children. Filled by
    /// `order`; walk it backwards for a post-order (`visitAfter`) pass.
    pub pre_order: Vec<u32>,
}

impl Quadtree {
    pub fn new() -> Quadtree {
        Quadtree {
            x0: Real::NAN,
            y0: Real::NAN,
            x1: Real::NAN,
            y1: Real::NAN,
            root: NONE,
            children: Vec::new(),
            next: Vec::new(),
            xs: Vec::new(),
            ys: Vec::new(),
            ix: Vec::new(),
            iy: Vec::new(),
            iv: Vec::new(),
            lv: Vec::new(),
            pre_order: Vec::new(),
        }
    }

    /// Rebuild the tree over `n` points (d3's `quadtree(nodes, x, y)`).
    /// Points with a non-finite coordinate are left out, as d3 leaves out NaN.
    pub fn build<C>(&mut self, n: usize, coord: C)
    where
        C: Fn(usize) -> (Real, Real),
    {
        self.x0 = Real::NAN;
        self.y0 = Real::NAN;
        self.x1 = Real::NAN;
        self.y1 = Real::NAN;
        self.root = NONE;
        self.children.clear();
        self.pre_order.clear();
        self.next.clear();
        self.next.resize(n, NONE);
        self.lv.clear();
        self.lv.resize(n, 0.0);
        self.xs.clear();
        self.ys.clear();

        let mut ex0 = Real::INFINITY;
        let mut ey0 = Real::INFINITY;
        let mut ex1 = Real::NEG_INFINITY;
        let mut ey1 = Real::NEG_INFINITY;
        for i in 0..n {
            let (x, y) = coord(i);
            self.xs.push(x);
            self.ys.push(y);
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            if x < ex0 {
                ex0 = x;
            }
            if x > ex1 {
                ex1 = x;
            }
            if y < ey0 {
                ey0 = y;
            }
            if y > ey1 {
                ey1 = y;
            }
        }
        if ex0 > ex1 || ey0 > ey1 {
            return; // no valid points
        }

        self.cover(ex0, ey0);
        self.cover(ex1, ey1);
        for i in 0..n {
            if self.xs[i].is_finite() && self.ys[i].is_finite() {
                self.add(i);
            }
        }
    }

    /// Grow the root square to contain a point (`cover.js`). Only called
    /// while the tree is still empty, so there is never a root to re-parent.
    fn cover(&mut self, x: Real, y: Real) {
        let (mut x0, mut y0, mut x1, mut y1) = (self.x0, self.y0, self.x1, self.y1);
        if x0.is_nan() {
            // Integer extent, so that doubling never moves a cell boundary.
            x0 = x.floor();
            x1 = x0 + 1.0;
            y0 = y.floor();
            y1 = y0 + 1.0;
        } else {
            let mut z = x1 - x0;
            if !(z != 0.0) {
                z = 1.0;
            }
            while x0 > x || x >= x1 || y0 > y || y >= y1 {
                let i = ((y < y0) as u8) << 1 | (x < x0) as u8;
                z *= 2.0;
                match i {
                    0 => {
                        x1 = x0 + z;
                        y1 = y0 + z;
                    }
                    1 => {
                        x0 = x1 - z;
                        y1 = y0 + z;
                    }
                    2 => {
                        x1 = x0 + z;
                        y0 = y1 - z;
                    }
                    _ => {
                        x0 = x1 - z;
                        y0 = y1 - z;
                    }
                }
                if !z.is_finite() {
                    break;
                }
            }
        }
        self.x0 = x0;
        self.y0 = y0;
        self.x1 = x1;
        self.y1 = y1;
    }

    fn new_internal(&mut self) -> u32 {
        self.children.push([NONE; 4]);
        (self.children.len() - 1) as u32
    }

    /// Insert point `i` (`add.js`).
    fn add(&mut self, i: usize) {
        let x = self.xs[i];
        let y = self.ys[i];
        let new_leaf = leaf(i);

        if self.root == NONE {
            self.root = new_leaf;
            return;
        }

        let (mut x0, mut y0, mut x1, mut y1) = (self.x0, self.y0, self.x1, self.y1);
        let mut node = self.root;
        let mut parent = NONE;
        let mut slot = 0usize;

        // Walk down to the leaf the new point lands on, or to an empty slot.
        while !is_leaf(node) {
            let xm = (x0 + x1) / 2.0;
            let ym = (y0 + y1) / 2.0;
            let right = x >= xm;
            let bottom = y >= ym;
            if right {
                x0 = xm;
            } else {
                x1 = xm;
            }
            if bottom {
                y0 = ym;
            } else {
                y1 = ym;
            }
            parent = node;
            slot = (bottom as usize) << 1 | right as usize;
            node = self.children[parent as usize][slot];
            if node == NONE {
                self.children[parent as usize][slot] = new_leaf;
                return;
            }
        }

        // Exactly coincident with the point already there: chain, newest first.
        let p = leaf_point(node);
        let xp = self.xs[p];
        let yp = self.ys[p];
        if x == xp && y == yp {
            self.next[i] = p as u32;
            self.replace(parent, slot, new_leaf);
            return;
        }

        // Otherwise split until the old and the new point part company.
        let mut depth = 0;
        loop {
            let k = self.new_internal();
            self.replace(parent, slot, k);
            parent = k;

            let xm = (x0 + x1) / 2.0;
            let ym = (y0 + y1) / 2.0;
            let right = x >= xm;
            let bottom = y >= ym;
            if right {
                x0 = xm;
            } else {
                x1 = xm;
            }
            if bottom {
                y0 = ym;
            } else {
                y1 = ym;
            }
            slot = (bottom as usize) << 1 | right as usize;
            let old_slot = ((yp >= ym) as usize) << 1 | (xp >= xm) as usize;
            if slot != old_slot {
                self.children[k as usize][old_slot] = node;
                self.children[k as usize][slot] = new_leaf;
                return;
            }
            depth += 1;
            if depth >= MAX_SPLIT_DEPTH {
                // Cannot be separated at this precision: treat as coincident.
                self.next[i] = p as u32;
                self.children[k as usize][slot] = new_leaf;
                return;
            }
        }
    }

    fn replace(&mut self, parent: u32, slot: usize, with: u32) {
        if parent == NONE {
            self.root = with;
        } else {
            self.children[parent as usize][slot] = with;
        }
    }

    /// Fill `pre_order` with every internal node, parents before children.
    pub fn order(&mut self) {
        self.pre_order.clear();
        if self.root == NONE || is_leaf(self.root) {
            return;
        }
        self.pre_order.push(self.root);
        let mut at = 0;
        while at < self.pre_order.len() {
            let k = self.pre_order[at] as usize;
            at += 1;
            for c in self.children[k] {
                if c != NONE && !is_leaf(c) {
                    self.pre_order.push(c);
                }
            }
        }
    }

    /// Pre-order traversal (`visit.js`). The callback gets a node reference
    /// and its square, and returns true to skip that node's children.
    #[inline]
    pub fn visit<F>(&self, stack: &mut Vec<Quad>, mut callback: F)
    where
        F: FnMut(u32, Real, Real, Real, Real) -> bool,
    {
        stack.clear();
        if self.root == NONE {
            return;
        }
        stack.push(Quad { node: self.root, x0: self.x0, y0: self.y0, x1: self.x1, y1: self.y1 });
        while let Some(q) = stack.pop() {
            if !callback(q.node, q.x0, q.y0, q.x1, q.y1) && !is_leaf(q.node) {
                let xm = (q.x0 + q.x1) / 2.0;
                let ym = (q.y0 + q.y1) / 2.0;
                let ch = &self.children[q.node as usize];
                if ch[3] != NONE {
                    stack.push(Quad { node: ch[3], x0: xm, y0: ym, x1: q.x1, y1: q.y1 });
                }
                if ch[2] != NONE {
                    stack.push(Quad { node: ch[2], x0: q.x0, y0: ym, x1: xm, y1: q.y1 });
                }
                if ch[1] != NONE {
                    stack.push(Quad { node: ch[1], x0: xm, y0: q.y0, x1: q.x1, y1: ym });
                }
                if ch[0] != NONE {
                    stack.push(Quad { node: ch[0], x0: q.x0, y0: q.y0, x1: xm, y1: ym });
                }
            }
        }
    }
}
