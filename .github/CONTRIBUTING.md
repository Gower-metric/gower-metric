# Contributing to gower-metric library

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> [!NOTE]
> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues
> - Cite the project in your publications


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Local Setup](#local-setup)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Join The Project Team](#join-the-project-team)


## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](./CODE_OF_CONDUCT.md).

## Local Setup

To set up your local development environment, it is highly recommended to use following tools:
- Python 3.10 or higher
- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/)
- [pre-commit](https://pre-commit.com/)
- [tox](https://tox.readthedocs.io/en/latest/)

After installing the required tools and checking out [this](#i-want-to-contribute) section:
1. Fork the repository
2. Create a new branch for your changes
3. Add new features or fix bugs
4. Make sure to run tests and linters before committing your changes
5. Commit your changes with a clear and descriptive message

### Installing Dependencies

In total, you need to know about following dependency groups:
- `base` - These dependencies are required to run the project
- `dev` - These dependencies are required for development
- `tests` - These dependencies are required to run the tests
- `r_env_tests` - These dependencies are required to run the R environment tests
- `docs` - These dependencies are required to build the documentation
- `benchmark` - These dependencies are required to run the benchmarks

You can install desired dependency using following command:

```bash
uv sync --group dev
```
This will install all dependencies from `base` and `dev` group. You can replace `dev` with any other group name to install dependencies from that group.

### Running Tests

To run the tests, you can use `tox`. This will create a virtual environment and run the tests in it. You can run all tests using following command:

```bash
tox -e py310
```
Where `py310` is the python version you want to use. You can also run tests for other python versions by replacing `py310` with `py311`, `py312`, etc. You can also run tests for all python versions using following command:

```bash
tox -r -e py310,py311,py312,py313,py314
```

> [!NOTE]
> The key difference between `tests` and `r_env_tests` is presence of [rpy2](https://rpy2.readthedocs.io/en/latest/) package, which requires [R](https://www.r-project.org) to be installed on your system. If you want to run tests that do not require R, feel free to use first group.

### Test Coverage

Coverage is collected automatically every time you run `uv run pytest` — no extra flags needed. The report shows which lines are missing coverage right in the terminal:

```bash
uv run pytest
```

At the end of the output you'll see a table like this:

```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
gower_metric/core/config.py          95      3    97%   122, 155, 180
gower_metric/core/metric.py         210     12    94%   289-295, 400
...
```

The `Missing` column tells you exactly which lines still need test coverage. This is powered by `pytest-cov` with the `--cov-report=term-missing` flag configured in `pyproject.toml`.

> [!IMPORTANT]
> When contributing, please make sure your changes **do not decrease** the overall test coverage. If you're adding new functionality, include tests that cover the new code paths. If you're modifying existing code, verify that the existing tests still pass and that the coverage for the affected files remains the same or improves.

### Creating documentation

To create the documentation, you can use [sphinx-autobuild](https://github.com/sphinx-doc/sphinx-autobuild) tool. It will automatically create the documentation for you and host it locally. You can run the following command to create the environment first:

```bash
uv sync --group docs
```

After that, you can run the following command to start the documentation server from the root directory of the project:

```bash
sphinx-autobuild docs/source docs/build/html
```

## I Have a Question

> [!NOTE]
> If you want to ask a question, we assume that you have read the available [documentation](https://gower-metric.readthedocs.io/en/latest/)

Before you ask a question, it is best to search for existing issues that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open a new issue.
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.



## I Want To Contribute

> [!NOTE]
> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs


#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://gower-metric.readthedocs.io/en/latest/). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
- Stack trace (Traceback)
- OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
- Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
- Possibly your input and the output
- Can you reliably reproduce the issue? And can you also reproduce it with older versions?


#### How Do I Submit a Good Bug Report?

> [!IMPORTANT]
> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Please contact one of the maintainers directly.


We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open a new issue. (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other tags (such as `critical`), and the issue will be left to be implemented by someone.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for `CONTRIBUTING.md`, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://gower-metric.readthedocs.io/en/latest/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a search among issues to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.


#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as GitHub issues.

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) on Linux. 
- **Explain why this enhancement would be useful** to most contributing users. You may also want to point out the other projects that solved it better and which could serve as inspiration.


### Improving The Documentation

You are welcome to help us improve the documentation, be it fixing typos, improving phrasing or adding new sections. Just open a pull request with your proposed changes. If you want to add a new section, please open an issue first to discuss the content and scope of the new section. What kind of tools and setup you need to build the documentation locally is described in the [README.md](./README.md) file.


## Join The Project Team

Just open an issue or contact one of the maintainers. We will then discuss your ideas and see how we can help each other.