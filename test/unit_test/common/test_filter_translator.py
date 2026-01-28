import unittest
from common.doc_store.doc_store_models import MetadataFilters, MetadataFilter
from common.doc_store.filter_translator import SQLFilterTranslator, ESFilterTranslator


class TestFilterTranslators(unittest.TestCase):
    def test_sql_translator_normalization_and_security(self):
        """Test SQL translator condition normalization and basic security checks."""
        translator = SQLFilterTranslator()

        # Test basic translation
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator="==")], condition="and")
        sql, params = translator.translate(filters)
        self.assertEqual(sql, "age = %s")
        self.assertEqual(params, [25])

        # Test multiple filters and joiner normalization
        # INTENTIONAL PADDING: " or  " is used to verify normalization logic (trimming)
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator=">"), MetadataFilter(key="name", value="John", operator="==")], condition=" or  ")
        sql, params = translator.translate(filters)
        self.assertEqual(sql, "age > %s OR name = %s")
        self.assertEqual(params, [25, "John"])

        # Test IN operator with list
        filters = MetadataFilters(filters=[MetadataFilter(key="id", value=[1, 2, 3], operator="in")])
        sql, params = translator.translate(filters)
        self.assertEqual(sql, "id IN (%s, %s, %s)")
        self.assertEqual(params, [1, 2, 3])

        # Test SQL injection in key
        filters = MetadataFilters(filters=[MetadataFilter(key="age; DROP TABLE users", value=25, operator="==")])
        with self.assertRaisesRegex(ValueError, "Invalid identifier"):
            translator.translate(filters)

        # Test unsupported operator
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator="invalid")])
        with self.assertRaisesRegex(ValueError, "Unsupported operator"):
            translator.translate(filters)

    def test_es_translator_operators_and_conditions(self):
        """Test ES translator with various operators, conditions, and boundary values."""
        translator = ESFilterTranslator()

        # Test basic term and range operators
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="status", value="active", operator="=="),
                MetadataFilter(key="price", value=100, operator="<="),
                MetadataFilter(key="category", value="electronics", operator="!="),
            ]
        )
        result = translator.translate(filters)
        # Expected: [{"term": {"status": "active"}}, {"range": {"price": {"lte": 100}}}, {"bool": {"must_not": [{"term": {"category": "electronics"}}]}}]
        self.assertEqual(len(result), 3)
        self.assertIn({"term": {"status": "active"}}, result)
        self.assertIn({"range": {"price": {"lte": 100}}}, result)
        self.assertIn({"bool": {"must_not": [{"term": {"category": "electronics"}}]}}, result)

        # Test OR condition
        filters = MetadataFilters(filters=[MetadataFilter(key="a", value=1, operator="=="), MetadataFilter(key="b", value=2, operator="==")], condition="OR")
        result = translator.translate(filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bool"]["should"], [{"term": {"a": 1}}, {"term": {"b": 2}}])

        # Test range validation
        filters = MetadataFilters(filters=[MetadataFilter(key="price", value={"invalid": 10}, operator="range")])
        with self.assertRaisesRegex(ValueError, "Invalid range operator"):
            translator.translate(filters)

        filters = MetadataFilters(filters=[MetadataFilter(key="price", value=10, operator="range")])
        with self.assertRaisesRegex(TypeError, "must be a dict"):
            translator.translate(filters)

        # Test unsupported operator
        filters = MetadataFilters(filters=[MetadataFilter(key="status", value="active", operator="invalid_op")])
        with self.assertRaisesRegex(ValueError, "Unsupported ES operator"):
            translator.translate(filters)

        # Test additional operators and boundary values
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="v1", value=0, operator="=="),
                MetadataFilter(key="v2", value=-1, operator=">"),
                MetadataFilter(key="v3", value=999999, operator="<"),
                MetadataFilter(key="v4", value=None, operator="=="),  # Should handle None gracefully if translator supports it or skip/error
            ]
        )
        # Note: Current translation might not strictly validate None, but let's just ensure it runs.
        # If None is invalid, we expect an error or handled dict.
        # Assuming ES translator handles simple formatting.
        result = translator.translate(filters)
        self.assertEqual(len(result), 4)

        # Test GT/LT/GTE/LTE
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="x", value=10, operator=">"),
                MetadataFilter(key="y", value=20, operator="<"),
                MetadataFilter(key="z", value=30, operator=">="),
                MetadataFilter(key="w", value=40, operator="<="),
            ]
        )
        result = translator.translate(filters)
        self.assertEqual(len(result), 4)
        self.assertIn({"range": {"x": {"gt": 10}}}, result)
        self.assertIn({"range": {"y": {"lt": 20}}}, result)
        self.assertIn({"range": {"z": {"gte": 30}}}, result)
        self.assertIn({"range": {"w": {"lte": 40}}}, result)

        # Test empty filters
        filters = MetadataFilters(filters=[])
        result = translator.translate(filters)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
