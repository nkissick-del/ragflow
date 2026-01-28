import unittest
from common.doc_store.doc_store_models import MetadataFilters, MetadataFilter
from common.doc_store.filter_translator import SQLFilterTranslator, ESFilterTranslator


class TestFilterTranslators(unittest.TestCase):
    def test_sql_translator_normalization_and_security(self):
        translator = SQLFilterTranslator()

        # Test basic translation
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator="==")], condition="and")
        self.assertEqual(translator.translate(filters), "age = 25")

        # Test multiple filters and joiner normalization
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator=">"), MetadataFilter(key="name", value="John", operator="==")], condition=" or  ")
        self.assertEqual(translator.translate(filters), "age > 25 OR name = 'John'")

        # Test SQL injection in key
        filters = MetadataFilters(filters=[MetadataFilter(key="age; DROP TABLE users", value=25, operator="==")])
        with self.assertRaisesRegex(ValueError, "Invalid identifier"):
            translator.translate(filters)

        # Test unsupported operator
        filters = MetadataFilters(filters=[MetadataFilter(key="age", value=25, operator="invalid")])
        with self.assertRaisesRegex(ValueError, "Unsupported operator"):
            translator.translate(filters)

    def test_es_translator_operators_and_conditions(self):
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


if __name__ == "__main__":
    unittest.main()
