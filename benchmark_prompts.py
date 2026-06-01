"""Curated benchmark prompts (Issue #24), 5 per testable category.

ASCII-only. The two objective categories are phrased to match success.py's
validators (structured_output -> must ask for JSON; modification_or_edit ->
must request Python / a diff), so the benchmark measures capability, not
whether the model guessed the output format. See the spec's "validator
coupling" section.
"""

# A reusable ~200-word passage for long_context (needle-in-haystack prompts).
_LONG_PASSAGE = (
    "The Aurelia ingestion service receives telemetry from field sensors over "
    "MQTT and writes batches to a column store every thirty seconds. Each batch "
    "is keyed by a region code and a monotonically increasing sequence number. "
    "The service listens on port 8472 and exposes a health endpoint at /healthz "
    "that returns the last committed sequence number. Operators configure the "
    "batch window with the AURELIA_FLUSH_SECONDS environment variable, which "
    "defaults to thirty. When the column store is unreachable, Aurelia buffers "
    "up to ten thousand records in memory and then applies backpressure to the "
    "MQTT subscription, pausing new reads until the store recovers. A separate "
    "compaction job runs nightly at 02:00 UTC and merges small segments into "
    "daily partitions, dropping any record older than ninety days per the "
    "retention policy. The compaction job reports its progress to the same "
    "metrics endpoint used by the ingestion path, tagged with the label "
    "job=compaction so dashboards can separate steady-state ingestion from the "
    "nightly maintenance spike. Region codes are three uppercase letters; an "
    "unknown region code causes the batch to be routed to a dead-letter topic "
    "named aurelia.deadletter for manual inspection the next business day."
)


def _ctx(question):
    return _LONG_PASSAGE + "\n\nQuestion: " + question


PROMPTS = {
    "quick_question": [
        "What is the time complexity of binary search?",
        "What does the Python enumerate function do?",
        "What is the difference between a list and a tuple in Python?",
        "Which HTTP status code means Not Found?",
        "What is the default TCP port for HTTPS?",
    ],
    "explanation_or_summary": [
        "Explain how a hash map achieves average constant-time lookups.",
        "Summarize what the SOLID principles are in software design.",
        "Explain the difference between a process and a thread.",
        "Describe what a database index does and why it speeds up queries.",
        "Explain what the GIL is in CPython and why it matters for threads.",
    ],
    "structured_output": [
        "Return a JSON object describing a fictional book with keys title, "
        "author, year, and genre.",
        "Output a JSON array of three fictional users, each an object with keys "
        "id, name, and email.",
        "Give me a single JSON object for a TODO item with keys id, text, done, "
        "and priority.",
        "Return only valid JSON: an object with keys city, country, and "
        "population for a fictional city.",
        "Produce a JSON object representing an HTTP request with keys method, "
        "url, headers (an object), and body.",
    ],
    "analytical_task": [
        "Compare and contrast REST and GraphQL for a public API.",
        "Evaluate the trade-offs between SQL and NoSQL databases for a social "
        "feed.",
        "Compare optimistic and pessimistic locking, giving one use case each.",
        "Assess the pros and cons of microservices versus a monolith for a "
        "small team.",
        "Contrast depth-first and breadth-first search and note when each is "
        "preferable.",
    ],
    "creative_generation": [
        "Write a short product tagline for a privacy-first local AI tool.",
        "Draft a two-sentence release announcement for a new CLI feature.",
        "Write a brief, friendly onboarding welcome message for a developer "
        "tool.",
        "Compose a short limerick about debugging code late at night.",
        "Write a one-paragraph elevator pitch for a note-taking app for "
        "engineers.",
    ],
    "modification_or_edit": [
        "Here is a Python function:\n\ndef add(a, b):\n    return a + b\n\n"
        "Modify it to validate that both arguments are numbers, raising "
        "TypeError otherwise. Return the full updated function.",
        "Given this Python:\n\ndef read_file(path):\n    return open(path)."
        "read()\n\nRewrite it to use a context manager and return an empty "
        "string if the file is missing. Return the full Python function.",
        "Refactor this Python to use a list comprehension and return the full "
        "function:\n\ndef evens(nums):\n    out = []\n    for n in nums:\n"
        "        if n % 2 == 0:\n            out.append(n)\n    return out",
        "Add a docstring and type hints to this Python function and return the "
        "full updated version:\n\ndef greet(name):\n    return 'Hello ' + name",
        "Modify this Python function to cache results with "
        "functools.lru_cache and return the full function:\n\ndef fib(n):\n"
        "    return n if n < 2 else fib(n - 1) + fib(n - 2)",
    ],
    "multi_step_or_planning": [
        "Outline a step-by-step plan to add rate limiting to an existing REST "
        "API.",
        "Give a numbered plan to migrate a project from REST to GraphQL "
        "incrementally.",
        "Plan the steps to set up continuous integration for a Python project, "
        "from zero to a green build.",
        "Outline a phased plan to add offline support to a web application.",
        "Provide a step-by-step plan to debug a memory leak in a long-running "
        "service.",
    ],
    "long_context": [
        _ctx("What port does the Aurelia service listen on?"),
        _ctx("Which environment variable configures the batch window, and what "
             "is its default?"),
        _ctx("What happens when the column store is unreachable?"),
        _ctx("When does the compaction job run and what does it drop?"),
        _ctx("Where is a batch routed if its region code is unknown?"),
    ],
}
