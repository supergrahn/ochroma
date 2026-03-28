# Building Ochroma

## Linux (primary)
```bash
cargo build --release --bin ochroma
```

## Windows Cross-Compilation
```bash
# Install Windows target
rustup target add x86_64-pc-windows-gnu
sudo apt install gcc-mingw-w64-x86-64

# Build
cargo build --release --target x86_64-pc-windows-gnu --bin ochroma
```

## Windows Native (on Windows)
```bash
cargo build --release --bin ochroma
```

## Release Package
```bash
./scripts/package.sh
```
