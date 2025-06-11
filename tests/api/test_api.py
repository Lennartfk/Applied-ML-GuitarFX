import os
import unittest
from fastapi.testclient import TestClient
from GuitarFX.api.api import app

client = TestClient(app)


class TestAPI(unittest.TestCase):
    """
    Integration test for the /predict endpoint for the guitar effect classifier
    API.
    """
    def setUp(self):
        """"
        Sets up the TestAPI class to add a path to the testing audio file and
        checks whether the path exists to the testing audio file.
        """
        self.test_audio_path = "tests/distorted-guitar-sustained" + \
                               "-to-chord_66bpm_C_major.wav"
        self.assertTrue(os.path.exists(self.test_audio_path), "Can't find " +
                        f"path to test audio: {self.test_audio_path}")

    def test_predict_audio_path(self):
        """
        Integration test on the /predict endpoint for the guitar effect
        classifier API.

        Included tests are:
            - Check if the request on the "/predict" endpoint works using an
            audio file.
            - Check if the response from the "/predict" endpoint
            returns the correct json structure.
        """
        with open(self.test_audio_path, "rb") as audio_file:
            response = client.post(
                "/predict",
                files={"audio_files": ("test_audio.wav", audio_file,
                                       "audio/wav")}
            )

        self.assertEqual(response.status_code, 200, "Status error:" +
                         f"{response.status_code}")

        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertIsInstance(response_data["predictions"], list)

        first_prediction = response_data["predictions"][0]
        self.assertIn("file_name", first_prediction)
        self.assertIn("confidences", first_prediction)
        self.assertIsInstance(first_prediction["confidences"], list)

        first_confidence = first_prediction["confidences"][0]
        self.assertIn("effect", first_confidence)
        self.assertIn("confidence", first_confidence)
        self.assertIsInstance(first_confidence["confidence"], float)


if __name__ == "__main__":
    unittest.main()
