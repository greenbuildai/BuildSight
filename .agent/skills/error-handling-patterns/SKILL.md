---
name: error-handling-patterns
description: Master error handling patterns across languages including exceptions, Result types, error propagation, and graceful degradation to build resilient applications. Use when implementing error handling, designing APIs, or improving application reliability.
---

# Error Handling Patterns

**Build resilient applications with robust error handling strategies that gracefully handle failures and provide excellent debugging experiences.**

## When to Use This Skill
- Implementing error handling in new features
- Designing error-resilient APIs
- Debugging production issues
- Improving application reliability
- Creating better error messages for users and developers
- Implementing retry and circuit breaker patterns
- Handling async/concurrent errors
- Building fault-tolerant distributed systems

## Core Philosophies

### Exceptions vs Result Types

*   **Exceptions**: Traditional try-catch, disrupts control flow. Best for unexpected errors or "exceptional" conditions.
*   **Result Types**: Explicit success/failure, functional approach. Best for expected errors and validation failures.
*   **Panics/Crashes**: Reserved for unrecoverable errors and programming bugs.

### Error Categories

*   **Recoverable**: Network timeouts, missing files, invalid user input, API rate limits.
*   **Unrecoverable**: Out of memory, stack overflow, programming bugs (null pointer, etc.).

---

## Language-Specific Implementation Guides

### [Python Patterns](patterns/python.md)
*   Custom Exception Hierarchies
*   Context Managers for Cleanup
*   Exponential Backoff Retries

### [TypeScript/JavaScript Patterns](patterns/typescript.md)
*   Custom Error Classes
*   Result Type Pattern
*   Async Error Handling

### [Rust Patterns](patterns/rust.md)
*   Result and Option Types
*   Combining Option and Result

### [Go Patterns](patterns/go.md)
*   Explicit Error Returns
*   Error Wrapping and Unwrapping

---

## Universal Reliability Patterns

### 1. Circuit Breaker
Prevents cascading failures in distributed systems by "opening" the circuit when failures reach a threshold.
[View Implementation Details](patterns/universal.md#circuit-breaker)

### 2. Error Aggregation
Collects multiple errors (e.g., form validation) instead of failing on the first one, providing a complete report to the user.
[View Implementation Details](patterns/universal.md#error-aggregation)

### 3. Graceful Degradation
Provides fallback functionality (e.g., cached data, default values) when primary operations fail.
[View Implementation Details](patterns/universal.md#graceful-degradation)

---

## Best Practices Checklist

1.  **Fail Fast**: Validate input early, fail quickly.
2.  **Preserve Context**: Include stack traces, metadata, and timestamps.
3.  **Meaningful Messages**: Explain *what* happened and *how* to fix it.
4.  **Log Appropriately**: Error = log; Expected failure = don't spam.
5.  **Clean Up**: Always use `try-finally`, `defer`, or context managers.
6.  **Don't Swallow**: Log or re-throw; never silently ignore.
7.  **Type-Safety**: Use typed errors whenever language features allow.

## Resources
- [Exception Hierarchy Design](references/exception-hierarchy-design.md)
- [Error Recovery Strategies](references/error-recovery-strategies.md)
- [Async Error Handling](references/async-error-handling.md)
