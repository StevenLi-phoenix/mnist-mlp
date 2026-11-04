# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-11-04

### Added
- Three-hidden-layer MLP (784-512-256-128-10) trained on MNIST with a 55k/5k train/validation split and a single final test evaluation (97.95 %).
- Epoch selection on validation accuracy and `split.npz` with the exact split indices.
- Data exploration figures: class balance per split, pixel statistics, ink per digit, class means, random samples.
- Result figures: loss and validation curves, confusion matrix, top-5 confusion examples, confidence histogram, first-layer filters, per-class metrics.
- Six-page LaTeX report (`report/report.tex`, compiled PDF) structured as a paper and slide template.
