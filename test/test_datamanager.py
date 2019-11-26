import os
import tempfile

from styletf.data.datamanager import DataManager, content_url, style_url


def test_resize_img():

    with tempfile.TemporaryDirectory() as tmpdir:
        datamgr = DataManager()
        file_name = tmpdir + "/content.jpg"
        path = datamgr.download_image(file_name=file_name, url=content_url)
        resized = datamgr.resize_image(path)

        assert resized.shape == (224, 224, 3)