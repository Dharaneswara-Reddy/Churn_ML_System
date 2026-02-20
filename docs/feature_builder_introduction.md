# Centralized Feature Builder Introduction

## Context
Feature preprocessing logic existed in multiple files.

## Problem
Independent preprocessing increased risk of silent feature mismatch.

## Solution
A centralized feature builder module was introduced.

Both:
- training pipeline
- inference pipeline

now use the same feature preparation logic.

## Result
- single source of truth for features
- reduced maintenance complexity
- consistent model inputs

## Key Lesson
Feature engineering must be centralized to prevent training–serving skew.
