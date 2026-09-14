---
name: Clean Code
description: Enforces clean, readable, maintainable code and disciplined commenting in any language. Use when writing, modifying, refactoring, or reviewing code, or when asked to clean up a codebase.
---

# Clean Code

**Core rule:** clarity, simplicity, maintainability and correctness beat cleverness and line count. Code is read far more often than it is written, and much of it is now read by people who did not write or prompt it.

## How to apply

1. **Learn the local conventions first.** Read the surrounding code, the language's official style guide (PEP 8, Effective Go, Rust API Guidelines, the Google style guides, …), and any linter/formatter config in the repo. Where they conflict: a mandatory style guide or config wins, then the repository's established pattern, then the rules below.
2. **Inspect before adding.** Before changing a file, look at the code you touch and its immediate neighbours for the violations listed below.
3. **Fix what the task touches.** Refactor violations in the code you are modifying or that directly block the change. Do not build new code on top of a poor pattern, and do not copy a poor pattern because it already exists.
4. **Stay in scope.** Do not refactor unrelated code. Mention unrelated violations instead of fixing them.
5. **Preserve behaviour.** Refactors must not change observable behaviour unless the task requires it. Run the existing tests before and after; add a test first when refactoring untested logic.
6. **Keep refactors separable.** Keep pure cleanups distinguishable from functional changes (separate commits or clearly separate hunks) so they can be reviewed independently.

## Naming

- Names reveal intent: what it is or does and why it exists, not how it is implemented. `elapsed_days`, not `d`; `fetch_invoice`, not `do_stuff`.
- Follow the language's casing and idioms (`snake_case` in Python, `camelCase` in JS/Java, `PascalCase` types, `SCREAMING_CASE` constants where idiomatic).
- Functions and methods are verbs or verb phrases; classes and types are nouns; booleans read as predicates (`is_ready`, `has_contact`).
- Length matches scope: a loop index may be `i`; a module-level value needs a descriptive name. Don't pad names with redundant context (`User.user_name` → `User.name`) or type (`name_str`).
- One word per concept across the repository — don't mix `fetch`/`get`/`retrieve` for the same idea.
- Include units where ambiguity is possible (`timeout_s`, `force_n`, `width_mm`).
- Avoid abbreviations unless universal in the domain (`id`, `url`, `rgb`, `ik`).
- Projects, packages, directories, modules and files are named after what they contain, in the ecosystem's convention (e.g. lowercase package names in Python, kebab-case repos). No `utils`, `helpers`, `misc` or `common` dumping grounds when a specific name fits.

## Variables and state

- Declare in the narrowest scope, as close as possible to first use.
- Prefer immutable values (`const`, `final`, tuples, frozen dataclasses) and pure transformations over reassignment and in-place mutation, when it stays readable.
- Introduce a variable only when it names a meaningful intermediate result, avoids repeated work, or splits a hard-to-read expression. Inline variables that are used once and add no meaning.
- Replace magic numbers and strings with named constants when they carry meaning or appear more than once.
- No global mutable state unless the platform requires it; pass dependencies explicitly.

## Functions

- One function, one job, at one level of abstraction. If describing it needs "and", consider splitting.
- Small enough to understand at a glance, but don't fragment a clear linear sequence into many one-line helpers that force the reader to jump around.
- Few parameters (roughly ≤ 3–4). Group related parameters into a value object; avoid boolean flag parameters that switch behaviour — split into two functions instead.
- Shallow nesting: use guard clauses and early returns; extract deeply nested blocks.
- Make side effects obvious from the name, or avoid them. Separate queries (return data) from commands (change state) where practical.
- Return values rather than mutating inputs, unless mutation is the established idiom or a measured performance need.
- Handle errors at the boundary where something meaningful can be done. Validate external input (users, files, network, other systems); don't add defensive checks for states the code cannot reach.

## Classes and OOP

- Use a class when data and the behaviour that protects its invariants belong together. A stateless bag of functions is a module, not a class.
- **Encapsulation:** expose the minimum; keep internal state private; don't write getters/setters that just leak fields.
- **Composition over inheritance:** model "has a" / "uses a" / "can do" with composition. Use inheritance only for a genuine, stable "is a" relationship where subclasses honour the parent's contract (Liskov substitution).
- Keep hierarchies shallow (typically ≤ 2 levels of your own classes). Deep trees make behaviour hard to trace and change.
- **Polymorphism:** replace repeated type switches (`if kind == …` in several places) with polymorphic dispatch — but a single, local conditional is simpler than a class hierarchy.
- **Abstraction:** introduce an interface, base class or pattern only when there are at least two real implementations or a concrete, present need (e.g. a test seam). No abstraction for hypothetical future use.
- High cohesion, loose coupling: a class should change for one reason, and depend on narrow interfaces rather than concrete details of other classes.

## Simplicity

- Solve the current problem with the simplest clear solution. YAGNI: no speculative features, options, configuration, plugin systems or generality.
- Use design patterns only when they remove real complexity, never to demonstrate them.
- No spaghetti: clear data flow, no hidden coupling through shared state, no action at a distance, no control flow that jumps across modules.
- If a solution is hard to explain, simplify it before documenting it.

## Redundancy and conciseness

- Remove duplicated logic by extracting it once the duplication is real (roughly the third occurrence, or two copies that must always change together). Similar-looking code that changes for different reasons is not duplication.
- Delete dead code, unused imports, variables, parameters, commented-out code and obsolete branches that *your change* made unused; flag pre-existing dead code outside the task rather than silently deleting it.
- Prefer idiomatic, expressive constructs (comprehensions, `any`/`all`, destructuring, `with`/`using`, standard collection methods) when they are clearer than manual loops.
- Fewer lines is a goal only when readability is equal or better. Never compress logic into clever one-liners, nested ternaries or dense chains that a reader must decode.

## Built-ins and dependencies

- Prefer the language's built-ins and standard library, then the framework already in use, before writing your own or adding a package.
- Add a dependency only for a clear benefit (correctness, security, significant effort saved, a well-maintained standard solution). Weigh maintenance, licence, size and supply-chain risk. Pin versions per the repository's convention.
- Import only what is used; follow the repository's import style and ordering.

## Project structure and consistency

- Organise by feature or responsibility in the way the ecosystem expects (standard layout for the language/framework, tests where the test runner expects them).
- Keep module boundaries meaningful; avoid circular dependencies.
- Be consistent everywhere: formatting (use the project's formatter/linter), naming, error handling, logging and file layout. A consistent imperfect convention beats a mix of better ones.

## Maintainability

- Write code that is easy to test: pure logic separated from I/O, dependencies injected rather than hardcoded.
- Prefer explicit over implicit; make invariants visible through types, names and assertions where the language supports them.
- Leave the code you touch a little cleaner than you found it — within the task's scope.

## Comments

Prefer self-explanatory code. A comment is justified only when it adds information the code cannot express.

**Write a comment for:**
- **Why:** intent, the rationale for a surprising or unidiomatic choice, and alternatives rejected.
- **Constraints and warnings:** invariants, ordering or thread-safety requirements, performance traps, things that will break if changed.
- **External assumptions:** protocol, API, hardware or physics behaviour the code depends on, with a link to the reference.
- **Non-obvious correctness:** why an algorithm or tricky expression (e.g. a regex) is correct.
- **Bug fixes** that would otherwise look unnecessary: what broke and why this avoids it.
- **Sources:** a link to the origin of adapted code.
- **Actionable TODOs:** concrete and specific (what and why), not vague wishes.
- **Public API docs:** docstrings/doc comments for public modules, classes and functions, in the language's standard format, describing contract, parameters, return values and errors — not implementation.

**Do not write comments that:**
- Restate what the code does (`i += 1  # increment i`) or narrate line by line.
- Compensate for unclear names or tangled logic — rename or restructure instead. If a clear comment is hard to write, the code probably needs simplifying.
- Are speculative, conversational, or addressed to the reviewer ("fixed as requested", "hope this works").
- Reference the current task, ticket or change history ("added for feature X", "new version") — that belongs in commit messages and PRs, and rots.
- Leave commented-out code behind — version control keeps it.

**Keep comments healthy:**
- Concise — one sentence where possible — and placed next to the code they explain.
- Updated or deleted in the same change as the code they describe; an outdated comment is worse than none.
- When editing code, check the comments around it and remove any that are now redundant, wrong or noisy.

## Review checklist

Before finishing, check the changed code:
- [ ] Names reveal intent and follow the language and repository conventions.
- [ ] Each function does one thing; parameters and nesting are modest; side effects are obvious.
- [ ] No inheritance or abstraction without a present, concrete need.
- [ ] No duplicated, dead or commented-out code introduced or left behind by the change.
- [ ] Built-ins/standard library used where they fit; no unnecessary new dependencies or imports.
- [ ] Comments explain why, are accurate, and none merely restate the code.
- [ ] Behaviour preserved (tests pass), formatting and style consistent with the repository.

## Sources

- Google Engineering Practices, "What to look for in a code review": https://google.github.io/eng-practices/review/reviewer/looking-for.html
- Stack Overflow Blog, "Best practices for writing code comments": https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/
- Composition over inheritance: https://en.wikipedia.org/wiki/Composition_over_inheritance
- DataCamp, "Coding Best Practices: A Complete Guide (2026)": https://www.datacamp.com/tutorial/coding-best-practices-and-guidelines
- Language style guides: PEP 8 (https://peps.python.org/pep-0008/), Google Style Guides (https://google.github.io/styleguide/)
