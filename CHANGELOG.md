# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-04

### Added
- 13 AI agents for full app lifecycle management
- Phase 0: 6 planning agents (PRD, Tech Architecture, Coding Standards, DB Schema, API Contract, Task Decomposition)
- Phase 1-N: 6 build agents (Frontend, Backend, Database, API, Testing, DevOps)
- Orchestrator agent for scheduling and coordination
- Smart model routing with DeepSeek V4
- Graceful degradation when agents fail
- Conflict resolution between concurrent agents
- CLI interface for running the full pipeline
- Claude Code skill integration
- PyPI package distribution
- Example app templates (running tracker, campus food delivery, habit tracker, pet adoption)
- Comprehensive README with architecture documentation
- Chinese language README (README_CN.md)
- Project orchestrator for managing multi-agent workflows
- Skills system for extending agent capabilities

### Security
- Input validation on all agent outputs
- Sandboxed execution environment for generated code
