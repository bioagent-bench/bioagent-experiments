# Tasksets

Each subdirectory is an installable Verifiers v1 taskset package. A taskset owns data loading,
task setup/finalization, scoring, and task-level configuration. It must remain independent of the
agent harness so the same benchmark can later run under another compatible harness.
