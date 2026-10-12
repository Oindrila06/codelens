from codelens.diffs import analyze_patch, contiguous_ranges, nearest_commentable

PATCH = """@@ -1,4 +1,6 @@
 import os
+import sys
 
 def f():
-    return 1
+    x = 2
+    return x
@@ -20,3 +22,3 @@ class A:
     a = 1
-    b = 2
+    b = 3
     c = 4"""


def test_added_and_commentable_lines():
    info = analyze_patch(PATCH)
    assert info.added == {2, 5, 6, 23}
    assert {1, 2, 3, 4, 5, 6}.issubset(info.commentable)
    assert 22 in info.commentable and 24 in info.commentable
    assert 10 not in info.commentable
    assert info.hunks == [(1, 6), (22, 24)]


def test_removed_old_lines_tracked():
    assert analyze_patch(PATCH).removed_old == {4, 21}


def test_no_newline_marker_ignored():
    p = "@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n\\ No newline at end of file"
    info = analyze_patch(p)
    assert info.added == {1}


def test_empty_patch():
    assert analyze_patch(None).added == set()
    assert analyze_patch("").commentable == set()


def test_ranges_and_snapping():
    assert contiguous_ranges({3, 4, 5, 9}) == [(3, 5), (9, 9)]
    assert nearest_commentable(8, {5, 9}) == 9
    assert nearest_commentable(8, {1}) is None
    assert nearest_commentable(5, {5}) == 5
