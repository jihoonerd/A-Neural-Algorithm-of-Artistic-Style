import argparse

from styletf.data.data_utils import download_image, resize_image
from styletf.model.train import StyleTFTrain


def args_parse():
    parser = argparse.ArgumentParser(
        description="A Neural Algorithm of Artistic Style")
    parser.add_argument('--content_url', default="https://blog.hmgjournal.com/images_n/contents/170523_Gwanghwamun01.png",
                        help='Content image url')
    parser.add_argument('--style_url', default="https://www.theartstory.org/images20/works/van_gogh_vincent_7.jpg?2",
                        help='Style image url')
    parser.add_argument('--quick', action="store_true",
                        help="Set input image as the content image")
    parser.add_argument('--train_epochs', default=100000, type=int)
    parser.add_argument('--log_interval', default=1000,  type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = args_parse()

    content_file_name = "content.jpg"
    content_path = download_image(
        file_name=content_file_name, url=args.content_url)
    content_resized = resize_image(content_path)

    style_file_name = "style.jpg"
    style_path = download_image(file_name=style_file_name, url=args.style_url)
    style_resized = resize_image(style_path)

    content_layer = ['block4_conv2']
    style_layer = ['block1_conv1', 'block2_conv1',
                   'block3_conv1', 'block4_conv1', 'block5_conv1']

    trainer = StyleTFTrain(content_resized, style_resized, content_layer, style_layer,
                           quick=args.quick, train_epochs=args.train_epochs, log_interval=args.log_interval)
    trainer.train()
