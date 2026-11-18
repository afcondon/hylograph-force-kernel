//! Quadtree implementation for Barnes-Hut force calculation
//!
//! Direct port of D3's d3-quadtree for behavioral parity

/// A node in the quadtree - either internal (4 children) or leaf (point data)
#[derive(Clone)]
pub enum QuadNode {
    /// Internal node with 4 children (NW=0, NE=1, SW=2, SE=3)
    Internal {
        children: [Option<Box<QuadNode>>; 4],
        // Accumulated values for Barnes-Hut
        x: f32,      // Center of mass x
        y: f32,      // Center of mass y
        value: f32,  // Total strength
    },
    /// Leaf node with point data
    Leaf {
        index: usize,  // Index into positions array
        x: f32,        // Point x position
        y: f32,        // Point y position
        value: f32,    // Strength
        next: Option<Box<QuadNode>>,  // Linked list for coincident points
    },
}

impl QuadNode {
    fn new_internal() -> QuadNode {
        QuadNode::Internal {
            children: [None, None, None, None],
            x: 0.0,
            y: 0.0,
            value: 0.0,
        }
    }

    fn new_leaf(index: usize, x: f32, y: f32, strength: f32) -> QuadNode {
        QuadNode::Leaf {
            index,
            x,
            y,
            value: strength,
            next: None,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        matches!(self, QuadNode::Leaf { .. })
    }

    /// Get accumulated x (center of mass)
    pub fn get_x(&self) -> f32 {
        match self {
            QuadNode::Internal { x, .. } => *x,
            QuadNode::Leaf { x, .. } => *x,
        }
    }

    /// Get accumulated y (center of mass)
    pub fn get_y(&self) -> f32 {
        match self {
            QuadNode::Internal { y, .. } => *y,
            QuadNode::Leaf { y, .. } => *y,
        }
    }

    /// Get accumulated strength value
    pub fn get_value(&self) -> f32 {
        match self {
            QuadNode::Internal { value, .. } => *value,
            QuadNode::Leaf { value, .. } => *value,
        }
    }
}

/// Quadtree for spatial partitioning
pub struct Quadtree {
    pub root: Option<Box<QuadNode>>,
    pub x0: f32,  // Bounding box
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Quadtree {
    /// Create empty quadtree
    pub fn new() -> Quadtree {
        Quadtree {
            root: None,
            x0: f32::NAN,
            y0: f32::NAN,
            x1: f32::NAN,
            y1: f32::NAN,
        }
    }

    /// Build quadtree from positions and strengths
    pub fn from_nodes(positions: &[f32], strengths: &[f32], n_nodes: usize) -> Quadtree {
        let mut tree = Quadtree::new();

        // First pass: compute bounding box
        let mut x0 = f32::INFINITY;
        let mut y0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y1 = f32::NEG_INFINITY;

        for i in 0..n_nodes {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            if x.is_finite() && y.is_finite() {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }

        // Expand to square (D3 does this)
        let dx = x1 - x0;
        let dy = y1 - y0;
        if dx > dy {
            y1 = y0 + dx;
        } else {
            x1 = x0 + dy;
        }

        // Add small padding to avoid edge cases
        let padding = (x1 - x0).max(1.0) * 0.001;
        tree.x0 = x0 - padding;
        tree.y0 = y0 - padding;
        tree.x1 = x1 + padding;
        tree.y1 = y1 + padding;

        // Second pass: insert all points
        for i in 0..n_nodes {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            if x.is_finite() && y.is_finite() {
                let strength = if i < strengths.len() { strengths[i] } else { 1.0 };
                tree.add(i, x, y, strength);
            }
        }

        tree
    }

    /// Add a point to the quadtree
    pub fn add(&mut self, index: usize, x: f32, y: f32, strength: f32) {
        if x.is_nan() || y.is_nan() {
            return;
        }

        // Expand bounds if needed
        if self.x0.is_nan() {
            self.x0 = x;
            self.y0 = y;
            self.x1 = x + 1.0;
            self.y1 = y + 1.0;
        } else {
            self.cover(x, y);
        }

        let new_leaf = QuadNode::new_leaf(index, x, y, strength);

        if self.root.is_none() {
            self.root = Some(Box::new(new_leaf));
            return;
        }

        // Insert into tree
        self.insert(new_leaf, x, y);
    }

    /// Expand bounds to cover point
    fn cover(&mut self, x: f32, y: f32) {
        if x < self.x0 || x > self.x1 || y < self.y0 || y > self.y1 {
            // Expand bounds (simplified - D3 does doubling)
            self.x0 = self.x0.min(x);
            self.y0 = self.y0.min(y);
            self.x1 = self.x1.max(x);
            self.y1 = self.y1.max(y);

            // Make square
            let dx = self.x1 - self.x0;
            let dy = self.y1 - self.y0;
            if dx > dy {
                self.y1 = self.y0 + dx;
            } else {
                self.x1 = self.x0 + dy;
            }
        }
    }

    /// Insert a leaf node into the tree
    fn insert(&mut self, leaf: QuadNode, x: f32, y: f32) {
        let (leaf_x, leaf_y, leaf_strength, leaf_index) = match &leaf {
            QuadNode::Leaf { x, y, value, index, .. } => (*x, *y, *value, *index),
            _ => return,
        };

        // Navigate to insertion point
        let mut x0 = self.x0;
        let mut y0 = self.y0;
        let mut x1 = self.x1;
        let mut y1 = self.y1;

        let root = self.root.take();
        self.root = Some(Self::insert_recursive(
            root, leaf, leaf_x, leaf_y, leaf_index, x0, y0, x1, y1
        ));
    }

    fn insert_recursive(
        node: Option<Box<QuadNode>>,
        leaf: QuadNode,
        x: f32,
        y: f32,
        index: usize,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> Box<QuadNode> {
        match node {
            None => Box::new(leaf),
            Some(mut existing) => {
                match existing.as_mut() {
                    QuadNode::Internal { children, .. } => {
                        // Find which quadrant
                        let xm = (x0 + x1) / 2.0;
                        let ym = (y0 + y1) / 2.0;
                        let right = x >= xm;
                        let bottom = y >= ym;
                        let i = (bottom as usize) << 1 | (right as usize);

                        let (nx0, ny0, nx1, ny1) = Self::quadrant_bounds(x0, y0, x1, y1, i);

                        let child = children[i].take();
                        children[i] = Some(Self::insert_recursive(
                            child, leaf, x, y, index, nx0, ny0, nx1, ny1
                        ));
                        existing
                    }
                    QuadNode::Leaf { x: ex, y: ey, index: ei, value: ev, next } => {
                        // Check for coincident points
                        if (x - *ex).abs() < 0.0001 && (y - *ey).abs() < 0.0001 {
                            // Add to linked list
                            let mut new_leaf = leaf;
                            if let QuadNode::Leaf { next: ref mut n, .. } = new_leaf {
                                *n = Some(existing);
                            }
                            return Box::new(new_leaf);
                        }

                        // Split: create internal node
                        let old_leaf = QuadNode::Leaf {
                            index: *ei,
                            x: *ex,
                            y: *ey,
                            value: *ev,
                            next: next.take(),
                        };

                        let mut internal = QuadNode::new_internal();

                        // Insert both leaves
                        let xm = (x0 + x1) / 2.0;
                        let ym = (y0 + y1) / 2.0;

                        // Insert old leaf
                        let right_old = *ex >= xm;
                        let bottom_old = *ey >= ym;
                        let i_old = (bottom_old as usize) << 1 | (right_old as usize);

                        // Insert new leaf
                        let right_new = x >= xm;
                        let bottom_new = y >= ym;
                        let i_new = (bottom_new as usize) << 1 | (right_new as usize);

                        if let QuadNode::Internal { children, .. } = &mut internal {
                            let (nx0, ny0, nx1, ny1) = Self::quadrant_bounds(x0, y0, x1, y1, i_old);
                            children[i_old] = Some(Self::insert_recursive(
                                None, old_leaf, *ex, *ey, *ei, nx0, ny0, nx1, ny1
                            ));

                            if i_new != i_old {
                                let (nx0, ny0, nx1, ny1) = Self::quadrant_bounds(x0, y0, x1, y1, i_new);
                                children[i_new] = Some(Self::insert_recursive(
                                    None, leaf, x, y, index, nx0, ny0, nx1, ny1
                                ));
                            } else {
                                // Same quadrant - recurse
                                let (nx0, ny0, nx1, ny1) = Self::quadrant_bounds(x0, y0, x1, y1, i_new);
                                children[i_new] = Some(Self::insert_recursive(
                                    children[i_new].take(), leaf, x, y, index, nx0, ny0, nx1, ny1
                                ));
                            }
                        }

                        Box::new(internal)
                    }
                }
            }
        }
    }

    fn quadrant_bounds(x0: f32, y0: f32, x1: f32, y1: f32, quadrant: usize) -> (f32, f32, f32, f32) {
        let xm = (x0 + x1) / 2.0;
        let ym = (y0 + y1) / 2.0;
        match quadrant {
            0 => (x0, y0, xm, ym),   // NW
            1 => (xm, y0, x1, ym),   // NE
            2 => (x0, ym, xm, y1),   // SW
            3 => (xm, ym, x1, y1),   // SE
            _ => (x0, y0, x1, y1),
        }
    }

    /// Post-order traversal (visitAfter) - used to accumulate values up the tree
    pub fn visit_after<F>(&mut self, mut callback: F)
    where
        F: FnMut(&mut QuadNode, f32, f32, f32, f32),
    {
        if let Some(ref mut root) = self.root {
            Self::visit_after_recursive(root, &mut callback, self.x0, self.y0, self.x1, self.y1);
        }
    }

    fn visit_after_recursive<F>(
        node: &mut Box<QuadNode>,
        callback: &mut F,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) where
        F: FnMut(&mut QuadNode, f32, f32, f32, f32),
    {
        // First visit children (post-order)
        if let QuadNode::Internal { children, .. } = node.as_mut() {
            for i in 0..4 {
                if let Some(ref mut child) = children[i] {
                    let (cx0, cy0, cx1, cy1) = Self::quadrant_bounds(x0, y0, x1, y1, i);
                    Self::visit_after_recursive(child, callback, cx0, cy0, cx1, cy1);
                }
            }
        }
        // Then visit this node
        callback(node.as_mut(), x0, y0, x1, y1);
    }

    /// Pre-order traversal (visit) - used for Barnes-Hut force calculation
    /// Callback returns true to skip children
    pub fn visit<F>(&self, mut callback: F)
    where
        F: FnMut(&QuadNode, f32, f32, f32, f32) -> bool,
    {
        if let Some(ref root) = self.root {
            Self::visit_recursive(root, &mut callback, self.x0, self.y0, self.x1, self.y1);
        }
    }

    fn visit_recursive<F>(
        node: &Box<QuadNode>,
        callback: &mut F,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) where
        F: FnMut(&QuadNode, f32, f32, f32, f32) -> bool,
    {
        // Visit this node first
        let skip_children = callback(node.as_ref(), x0, y0, x1, y1);

        if !skip_children {
            if let QuadNode::Internal { children, .. } = node.as_ref() {
                for i in 0..4 {
                    if let Some(ref child) = children[i] {
                        let (cx0, cy0, cx1, cy1) = Self::quadrant_bounds(x0, y0, x1, y1, i);
                        Self::visit_recursive(child, callback, cx0, cy0, cx1, cy1);
                    }
                }
            }
        }
    }
}

/// Accumulate center of mass and total strength for Barnes-Hut
/// This is called via visitAfter
pub fn accumulate(node: &mut QuadNode, strengths: &[f32]) {
    match node {
        QuadNode::Internal { children, x, y, value, .. } => {
            let mut strength = 0.0f32;
            let mut weight = 0.0f32;
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;

            for child in children.iter() {
                if let Some(c) = child {
                    let c_value = c.get_value();
                    let c_weight = c_value.abs();
                    if c_weight > 0.0 {
                        strength += c_value;
                        weight += c_weight;
                        cx += c_weight * c.get_x();
                        cy += c_weight * c.get_y();
                    }
                }
            }

            if weight > 0.0 {
                *x = cx / weight;
                *y = cy / weight;
            }
            *value = strength;
        }
        QuadNode::Leaf { index, x: px, y: py, value, next, .. } => {
            // Leaf: accumulate from linked list
            let mut strength = if *index < strengths.len() {
                strengths[*index]
            } else {
                0.0
            };

            // Follow linked list
            let mut current = next.as_ref();
            while let Some(n) = current {
                if let QuadNode::Leaf { index: ni, next: nn, .. } = n.as_ref() {
                    if *ni < strengths.len() {
                        strength += strengths[*ni];
                    }
                    current = nn.as_ref();
                } else {
                    break;
                }
            }

            *value = strength;
            // x, y already set to point position
        }
    }
}
