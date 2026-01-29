import unittest
from common.doc_store.doc_store_base import OrderByExpr


class TestOrderByExpr(unittest.TestCase):
    def test_fields_immutability(self):
        order_by = OrderByExpr()
        order_by.asc("field1")

        fields = order_by.fields()
        self.assertIsInstance(fields, tuple)
        self.assertEqual(fields, (("field1", 0),))

        # Verify it's a copy/immutable and doesn't affect internal state if it were a list
        # Since it's a tuple, it's already immutable.
        # But let's check that appending to the original doesn't affect the returned tuple
        order_by.desc("field2")
        self.assertEqual(fields, (("field1", 0),))
        self.assertEqual(order_by.fields(), (("field1", 0), ("field2", 1)))

    def test_fields_prop(self):
        order_by = OrderByExpr()
        order_by.asc("field1")

        accessor = order_by.fields_prop
        self.assertEqual(len(accessor), 1)
        self.assertEqual(accessor[0], ("field1", 0))
        self.assertTrue(accessor)

        # Test __call__ returns tuple
        self.assertEqual(accessor(), (("field1", 0),))


if __name__ == "__main__":
    unittest.main()
