# Configuration Driven System

## Motivation

Hardcoded paths and parameters made the system fragile during refactoring.
Changing model locations or log behavior required editing multiple files.

## Approach

A centralized YAML configuration was introduced.

config/settings.yaml


Configuration controls:

- data paths
- experiment directories
- production model location
- inference thresholds
- logging outputs

## Benefits

- environment behavior can change without code edits
- consistent configuration across modules
- easier deployment adaptation

## Principle

Behavior that may change between environments should live in configuration,
not inside source code.
