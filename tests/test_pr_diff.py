"""pr_diff: annotating patches with real line numbers and fitting a PR into a budget."""

from pr_diff import annotate_patch, render_pr_diff

PATCH = """\
@@ -10,4 +10,5 @@ def f():
 a = 1
-b = 2
+b = 3
+c = 4
 d = 5
@@ -40,2 +45,3 @@ def g():
 x = 1
+y = 2
 z = 3
\\ No newline at end of file"""


def pr_file(name, patch="@@ -1,1 +1,2 @@\n a\n+b", add=1, dele=0, status="modified"):
    return {
        "filename": name,
        "status": status,
        "additions": add,
        "deletions": dele,
        "changes": add + dele,
        "patch": patch,
    }


# --- annotate_patch -------------------------------------------------------------


def test_annotate_numbers_added_and_context_lines_in_the_new_file():
    out = annotate_patch(PATCH).split("\n")
    assert out[0] == "@@ -10,4 +10,5 @@ def f():"
    assert out[1] == "   10  a = 1"  # context: new line 10
    assert out[2] == "      -b = 2"  # removed: no number
    assert out[3] == "   11 +b = 3"  # added: new line 11
    assert out[4] == "   12 +c = 4"
    assert out[5] == "   13  d = 5"


def test_numbering_restarts_at_each_hunk_header():
    out = annotate_patch(PATCH).split("\n")
    assert out[6] == "@@ -40,2 +45,3 @@ def g():"
    assert out[7] == "   45  x = 1"
    assert out[8] == "   46 +y = 2"
    assert out[9] == "   47  z = 3"


def test_no_newline_marker_is_preserved_without_consuming_a_line_number():
    out = annotate_patch(PATCH).split("\n")
    assert out[-1].strip() == "\\ No newline at end of file"


def test_trailing_newline_is_not_treated_as_a_blank_context_line():
    out = annotate_patch("@@ -1,1 +1,2 @@\n a\n+b\n").split("\n")
    assert out == ["@@ -1,1 +1,2 @@", "    1  a", "    2 +b"]


# --- render_pr_diff ---------------------------------------------------------------


def test_lockfiles_and_patchless_files_are_skipped_and_reported():
    files = [
        pr_file("src/app.py"),
        pr_file("package-lock.json", add=5000),
        {**pr_file("logo.png"), "patch": None},
    ]
    out = render_pr_diff(files)
    assert "=== src/app.py" in out
    assert "=== package-lock.json" not in out
    assert "package-lock.json (lock/generated file)" in out
    assert "logo.png" in out and "binary" in out


def test_source_is_reviewed_before_tests_before_docs():
    files = [
        pr_file("README.md", add=50),
        pr_file("tests/test_app.py", add=40),
        pr_file("src/app.py", add=1),
    ]
    out = render_pr_diff(files)
    assert (
        out.index("=== src/app.py")
        < out.index("=== tests/test_app.py")
        < out.index("=== README.md")
    )


def test_header_reports_totals_and_explains_the_line_column():
    out = render_pr_diff([pr_file("a.py", add=3, dele=1), pr_file("b.py", add=2)])
    assert "2 file(s) changed (+5 -1)" in out
    assert "NEW version of the file" in out


def test_a_long_file_is_truncated_with_a_pointer_to_get_the_rest():
    long_patch = "@@ -1,1 +1,400 @@\n" + "\n".join(f"+line {i}" for i in range(400))
    out = render_pr_diff([pr_file("big.py", patch=long_patch, add=400)], per_file_chars=800)
    assert "more diff lines omitted" in out
    assert "get_pr_diff with path='big.py'" in out


def test_total_budget_lists_files_that_did_not_fit():
    files = [
        pr_file(
            f"src/m{i}.py",
            patch="@@ -1,1 +1,40 @@\n" + "\n".join(f"+x{j}" for j in range(40)),
            add=40,
        )
        for i in range(8)
    ]
    out = render_pr_diff(files, max_chars=900, per_file_chars=800)
    assert "Not shown (size budget)" in out
    shown = out.count("=== src/m")
    assert 1 <= shown < 8


def test_at_least_one_file_is_always_shown_even_if_it_exceeds_the_budget():
    huge = "@@ -1,1 +1,900 @@\n" + "\n".join(f"+row {i}" for i in range(900))
    out = render_pr_diff(
        [pr_file("huge.py", patch=huge, add=900)], max_chars=1500, per_file_chars=1400
    )
    assert "=== huge.py" in out
    assert len(out) < 2600


def test_only_path_zooms_into_one_file_with_the_full_budget():
    files = [pr_file("a.py"), pr_file("pkg/b.py", patch="@@ -1,1 +1,2 @@\n x\n+only_here")]
    out = render_pr_diff(files, only_path="pkg/b.py")
    assert "only_here" in out and "=== a.py" not in out


def test_only_path_accepts_a_unique_suffix():
    out = render_pr_diff([pr_file("deep/dir/b.py")], only_path="b.py")
    assert "=== deep/dir/b.py" in out


def test_only_path_for_an_unknown_file_lists_what_did_change():
    out = render_pr_diff([pr_file("a.py"), pr_file("b.py")], only_path="nope.py")
    assert "not part of this PR" in out and "a.py" in out and "b.py" in out
