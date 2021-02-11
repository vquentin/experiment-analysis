# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)


class SEMZeissMetadata(object):
    """This class represents a SEM image's metadata. It has convenience
    functions to access the metadata and perform operations. Intended for
    images generated with Zeiss Scanning Electron Microscopes.

    Positional arguments:
    metadata: the metadata that needs to be parsed
    """
    def __init__(self, metadata, image_path):
        self.__raw_metadata = metadata
        self.image_path = image_path
        try:
            self.line_count = metadata["ap_line_counter"][1]
            self.pixel_size = metadata["ap_image_pixel_size"][1]
            self.beam_x_offset = metadata["ap_beam_offset_x"][1]
            self.beam_y_offset = metadata["ap_beam_offset_y"][1]
            self.stage_x = metadata["ap_stage_at_x"][1]
            self.stage_y = metadata["ap_stage_at_y"][1]
        except KeyError:
            log.exception(f"Could not read the metadata of \
                             {self.image_path.stem}. Is it a Zeiss image?")
            raise

    def __repr__(self):
        return f"{self.__class__.__name__}(<<metadata>>, {self.image_path})"

    def __str__(self):
        return (f"Metadata of image {self.image_path.stem} \n"
                f"\tLine count: {self.line_count}\n"
                f"\tPixel size: {self.pixel_size}")

    def write_to_file(self):
        """Write the raw metadata to a text file with the same name and
        directory than the associated image.
        """
        f = self.image_path.parent.joinpath(self.image_path.stem + '_meta.txt')
        with open(f, mode='w') as fid:
            print(json.dumps(self.__raw_metadata, indent=4), file=fid)

    def distance_to_image(self, *args):
        """Return distance between current image and other images.

        Positional arguments:
        *args: the image(s) to which the distance is calculated.
        """
        distances = [((self.stage_x-image.metadata.stage_x)**2
                      + (self.stage_y-image.metadata.stage_y)**2)**0.5
                     for image in args]
        return distances
