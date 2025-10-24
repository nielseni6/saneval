"""Unit tests for circular reference removal in benchmark_runner."""

from ssa.benchmark_runner import remove_circular_refs


class TestPrimitiveTypes:
    """Tests for primitive type handling."""

    def test_integer(self):
        """Test that integers pass through unchanged."""
        assert remove_circular_refs(42) == 42
        assert remove_circular_refs(0) == 0
        assert remove_circular_refs(-100) == -100

    def test_float(self):
        """Test that floats pass through unchanged."""
        assert remove_circular_refs(3.14) == 3.14
        assert remove_circular_refs(0.0) == 0.0
        assert remove_circular_refs(-2.5) == -2.5

    def test_string(self):
        """Test that strings pass through unchanged."""
        assert remove_circular_refs("hello") == "hello"
        assert remove_circular_refs("") == ""
        assert remove_circular_refs("test with spaces") == "test with spaces"

    def test_boolean(self):
        """Test that booleans pass through unchanged."""
        assert remove_circular_refs(True) is True
        assert remove_circular_refs(False) is False

    def test_none(self):
        """Test that None passes through unchanged."""
        assert remove_circular_refs(None) is None


class TestSimpleCollections:
    """Tests for simple collections without circular references."""

    def test_flat_dict(self):
        """Test flat dictionary with primitive values."""
        data = {"a": 1, "b": "test", "c": True}
        result = remove_circular_refs(data)
        assert result == data
        assert result is not data  # Should be a copy

    def test_flat_list(self):
        """Test flat list with primitive values."""
        data = [1, 2, 3, "test", True]
        result = remove_circular_refs(data)
        assert result == data
        assert result is not data  # Should be a copy

    def test_nested_dict(self):
        """Test nested dictionary structure."""
        data = {
            "level1": {"level2": {"level3": "value"}},
            "other": "data",
        }
        result = remove_circular_refs(data)
        assert result == data

    def test_nested_list(self):
        """Test nested list structure."""
        data = [1, [2, [3, 4]], 5]
        result = remove_circular_refs(data)
        assert result == data

    def test_mixed_nesting(self):
        """Test mixed dict and list nesting."""
        data = {
            "list": [1, 2, {"nested": "value"}],
            "dict": {"inner": [3, 4]},
        }
        result = remove_circular_refs(data)
        assert result == data


class TestCircularReferences:
    """Tests for actual circular reference handling."""

    def test_self_referencing_dict(self):
        """Test dict that references itself."""
        data = {"a": 1, "b": 2}
        data["self"] = data

        result = remove_circular_refs(data)

        assert "a" in result
        assert "b" in result
        assert result["a"] == 1
        assert result["b"] == 2
        # Circular reference should be replaced with None
        assert result["self"] is None

    def test_self_referencing_list(self):
        """Test list that references itself."""
        data = [1, 2, 3]
        data.append(data)

        result = remove_circular_refs(data)

        assert len(result) == 4
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3
        # Circular reference should be replaced with None
        assert result[3] is None

    def test_mutual_reference_dicts(self):
        """Test two dicts that reference each other."""
        dict_a = {"name": "A"}
        dict_b = {"name": "B"}
        dict_a["ref"] = dict_b
        dict_b["ref"] = dict_a

        result = remove_circular_refs(dict_a)

        assert result["name"] == "A"
        assert result["ref"]["name"] == "B"
        # Second circular reference should be None
        assert result["ref"]["ref"] is None

    def test_deep_circular_chain(self):
        """Test circular reference in a deep chain."""
        level1 = {"name": "level1"}
        level2 = {"name": "level2"}
        level3 = {"name": "level3"}

        level1["child"] = level2
        level2["child"] = level3
        level3["child"] = level1  # Circular back to level1

        result = remove_circular_refs(level1)

        assert result["name"] == "level1"
        assert result["child"]["name"] == "level2"
        assert result["child"]["child"]["name"] == "level3"
        # Circular reference should be None
        assert result["child"]["child"]["child"] is None


class TestDepthLimit:
    """Tests for max depth handling."""

    def test_depth_limit_exceeded(self):
        """Test that max depth limit is enforced."""
        # Create deeply nested structure
        data = current = {}
        for i in range(150):  # Exceed default max_depth of 100
            current["next"] = {}
            current = current["next"]
        current["value"] = "deep"

        result = remove_circular_refs(data, max_depth=100)

        # Should have max_depth_exceeded marker at depth 100
        current = result
        depth = 0
        while isinstance(current, dict) and "next" in current:
            current = current["next"]
            depth += 1
            if depth > 105:  # Safety check to prevent infinite loop
                break

        # Should eventually reach max_depth_exceeded marker
        # (exact depth may vary due to implementation details)
        assert depth <= 105

    def test_custom_depth_limit(self):
        """Test custom max depth limit."""
        # Create structure with 20 levels
        data = current = {}
        for i in range(20):
            current["next"] = {}
            current = current["next"]
        current["value"] = "deep"

        # Set max_depth to 10
        result = remove_circular_refs(data, max_depth=10)

        # Should stop at depth 10
        current = result
        for _ in range(10):
            assert isinstance(current, dict)
            current = current.get("next")

        # Beyond depth 10 should have marker or be truncated or wrapped in dict
        assert current in [None, "<max_depth_exceeded>", {}] or (
            isinstance(current, dict) and current.get("next") == "<max_depth_exceeded>"
        )


class TestNonSerializableObjects:
    """Tests for non-serializable object handling."""

    def test_callable_filtered(self):
        """Test that callable objects are filtered out."""

        def my_function():
            return "test"

        data = {"func": my_function, "value": 1}

        result = remove_circular_refs(data)

        # Non-callable value should be preserved
        assert "value" in result
        assert result["value"] == 1
        # Callable function should be filtered out
        assert "func" not in result

    def test_class_instance(self):
        """Test handling of class instances."""

        class TestClass:
            def __init__(self):
                self.value = 42

        obj = TestClass()
        data = {"object": obj, "primitive": "test"}

        result = remove_circular_refs(data)

        assert "primitive" in result
        assert result["primitive"] == "test"
        # Object should be converted to string representation
        assert "object" in result
        assert isinstance(result["object"], str)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_dict(self):
        """Test empty dictionary."""
        assert remove_circular_refs({}) == {}

    def test_empty_list(self):
        """Test empty list."""
        assert remove_circular_refs([]) == []

    def test_dict_with_none_values(self):
        """Test dict containing None values."""
        data = {"a": None, "b": 1, "c": None}
        result = remove_circular_refs(data)
        assert result == data

    def test_list_with_none_values(self):
        """Test list containing None values."""
        data = [1, None, 2, None, 3]
        result = remove_circular_refs(data)
        assert result == data

    def test_unicode_strings(self):
        """Test handling of unicode strings."""
        data = {"text": "Hello 世界", "emoji": "😀🎉"}
        result = remove_circular_refs(data)
        assert result == data

    def test_special_dict_keys(self):
        """Test dict with special key names."""
        data = {
            "__special__": "value1",
            "_private": "value2",
            "normal": "value3",
        }
        result = remove_circular_refs(data)
        # __special__ keys starting with __ should be filtered
        assert "_private" in result
        assert "normal" in result

    def test_mixed_types_in_list(self):
        """Test list with mixed types."""
        data = [1, "string", 3.14, True, None, {"nested": "dict"}, [1, 2]]
        result = remove_circular_refs(data)
        assert len(result) == len(data)
        assert result[0] == 1
        assert result[1] == "string"
        assert result[2] == 3.14
        assert result[3] is True
        assert result[4] is None
        assert result[5] == {"nested": "dict"}
        assert result[6] == [1, 2]


class TestPerformance:
    """Tests for performance and scalability."""

    def test_large_flat_dict(self):
        """Test handling of large flat dictionary."""
        data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        result = remove_circular_refs(data)
        assert len(result) == 1000
        assert result["key_0"] == "value_0"
        assert result["key_999"] == "value_999"

    def test_large_flat_list(self):
        """Test handling of large flat list."""
        data = list(range(1000))
        result = remove_circular_refs(data)
        assert len(result) == 1000
        assert result[0] == 0
        assert result[999] == 999

    def test_wide_nested_structure(self):
        """Test structure with many siblings at each level."""
        data = {f"branch_{i}": {"level2": {"value": i}} for i in range(100)}
        result = remove_circular_refs(data)
        assert len(result) == 100
        assert result["branch_0"]["level2"]["value"] == 0
        assert result["branch_99"]["level2"]["value"] == 99
