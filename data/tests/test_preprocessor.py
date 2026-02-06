import unittest
from src.preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_lowercase(self):
        raw_text = "HELLO World"
        clean_text = self.preprocessor.clean_text(raw_text)
        self.assertIn("hello", clean_text)
        self.assertNotIn("HELLO", clean_text)

    def test_remove_punctuation(self):
        raw_text = "Bad product!!! Not good, at all."
        clean_text = self.preprocessor.clean_text(raw_text)
        self.assertNotIn("!", clean_text)
        self.assertNotIn(",", clean_text)

    def test_empty_input(self):
        raw_text = ""
        clean_text = self.preprocessor.clean_text(raw_text)
        self.assertEqual(clean_text, "")

if __name__ == '__main__':
    unittest.main()
