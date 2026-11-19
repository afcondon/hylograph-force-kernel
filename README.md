# force-kernel

A force simulation kernel in Rust, built for WebAssembly. It is the optional
fast engine of
[hylograph-simulation](https://github.com/afcondon/purescript-hylograph-simulation),
whose default engine is the same forces written in PureScript
([hylograph-forces](https://github.com/afcondon/purescript-hylograph-forces)).

## What is in it

- **d3-force's forces**, ported from d3-force 3.0.0 over a port of
  d3-quadtree: centre, many-body, link, x, y, radial and collide, with d3's
  tick and its random source.
- **Six forces d3 does not have**: towards a target, rectangle collision,
  containment in a region, group cohesion, many-body scoped to a group, and
  a directional constraint on links. Each is defined by the doc comment on
  its `apply_*` function in `src/forces.rs`, precisely enough to reimplement
  from.
- Snapshot and restore of a whole simulation, and measures of whether it has
  settled.

The crate documentation at the head of `src/lib.rs` says what a tick does,
in order.

There is no source of nondeterminism in it: no clock, no hashing, no
threads, no callbacks into the host. The same calls give the same positions
on every run.

## Building

You need Rust with the `wasm32-unknown-unknown` target, and
[wasm-pack](https://rustwasm.github.io/wasm-pack/).

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

wasm-pack build --target web --out-dir pkg                     # the module as shipped, 32-bit floats
wasm-pack build --target web --out-dir pkg-f64 -- --features f64   # 64-bit, for checking only
```

`pkg/` then holds `force_kernel_bg.wasm` and its JavaScript loader. Last
built here with rustc 1.92.0 and wasm-pack 0.13.1.

`.envrc` is the author's own development environment (a Nix flake outside
this repository); ignore or delete it.

## Tests

```sh
cargo test                   # 127 tests, 32-bit
cargo test --features f64    # 63 tests, 64-bit
```

The tests hold every force to a plain, slow reference implementation in
`src/reference.rs`.

## What it agrees with, and how that is checked

- **Built with `--features f64`, the d3 forces give d3-force 3.0.0's
  positions bit for bit**, and all the forces give the PureScript kernel's
  positions bit for bit. The check lives in hylograph-forces
  (`test/differential.mjs`): thousands of generated scenarios, every value
  compared with exact equality, no tolerance.
- **The shipped build uses 32-bit floats and does not claim to match d3 or
  the PureScript kernel exactly.** It is held to the PureScript kernel in
  behaviour by hylograph-simulation's `test/engine-parity.mjs`.

## Licence

MIT, with d3-force's and d3-quadtree's ISC notice for the ported parts. See
`LICENSE`.
