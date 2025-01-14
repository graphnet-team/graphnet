"""More advanced example script to convert IceCube I3 files to sqlite."""

from glob import glob

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3TruthExtractor,
    I3ParticleExtractor,
    I3FeatureExtractorIceCube86,
    I3GenericExtractor,
    I3DictValueExtractor,
    I3FilterMapExtractor,
    I3TotalEExtractor,
    I3HighestEparticleExtractor,
)

from graphnet.data.extractors.icecube.utilities.i3_filters import (
    TableFilter,
    NullSplitI3Filter,
)

from graphnet.data.extractors.icecube.utilities.gcd_hull import GCD_hull

from graphnet.data.extractors.combine_extractors import CombinedExtractor


from graphnet.data.dataconverter import DataConverter
from graphnet.data.pre_configured import I3ToSQLiteConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)


def main(
    merge: bool,
    remove: bool,
    workers: int = 1,
    padding: int = 250,
    max_table_size: int = int(25e7),
) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    inputs = [f"{TEST_DATA_DIR}/i3/oscNext_muongun_level3_v02"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/ic86_advanced"
    gcd_rescue = glob(
        f"{TEST_DATA_DIR}/i3/oscNext_muongun_level3_v02/*GeoCalib*"
    )[0]

    # Create hulls, these need to be global for the multiprocessing.
    IceCube_hull = GCD_hull(gcd_rescue, padding=0)
    IceCube_hull_extended = GCD_hull(gcd_rescue, padding=padding)

    # Create Extractors
    # truth features
    truth = I3TruthExtractor(mctree="I3MCTree", exclude=["sim_type"])
    # energy features
    homogenized_qtot = I3GenericExtractor(
        keys=["Homogenized_QTot"], extractor_name="Homogenized_QTot"
    )
    E_daughter = I3TotalEExtractor(
        hull=IceCube_hull,
        mctree="I3MCTree",
        mmctracklist="MMCTrackList",
        extractor_name="Daughter",
        daughters=True,
    )
    E_Highest = I3TotalEExtractor(
        hull=IceCube_hull,
        mctree="I3MCTree",
        mmctracklist="MMCTrackList",
        extractor_name="Highest",
        daughters=False,
    )
    E_daughter_extended = I3TotalEExtractor(
        hull=IceCube_hull_extended,
        mctree="I3MCTree",
        mmctracklist="MMCTrackList",
        extractor_name="DaughterExtended" + str(padding),
        daughters=True,
    )
    E_Highest_extended = I3TotalEExtractor(
        hull=IceCube_hull_extended,
        mctree="I3MCTree",
        mmctracklist="MMCTrackList",
        extractor_name="HighestExtended" + str(padding),
        daughters=False,
    )

    # Combine all the "truth" like features into one extractor,
    # such that they can be written to the same table.

    combined_extractor = CombinedExtractor(
        extractors=[
            truth,
            homogenized_qtot,
            E_daughter,
            E_Highest,
            E_daughter_extended,
            E_Highest_extended,
        ],
        extractor_name="truth",
    )

    # Detector readout features
    SRTInIcePulses = I3FeatureExtractorIceCube86("SRTInIcePulses")

    # Particle extractors
    highest_e_in_ice = I3HighestEparticleExtractor(
        extractor_name="HighestEInIceParticle", hull=IceCube_hull
    )
    highest_e_daughter = I3HighestEparticleExtractor(
        extractor_name="HighestEDaughter", hull=IceCube_hull, daughters=True
    )
    highest_e_in_ice_extended = I3HighestEparticleExtractor(
        extractor_name="HighestEInIceParticleExtended" + str(padding),
        hull=IceCube_hull_extended,
    )
    highest_e_daughter_extended = I3HighestEparticleExtractor(
        extractor_name="HighestEDaughterExtended" + str(padding),
        hull=IceCube_hull_extended,
        daughters=True,
    )

    # Comparisons to other reconstruction methods

    SplineMPE = I3ParticleExtractor(extractor_name="OnlineL2_SplineMPE")
    LineFit = I3ParticleExtractor(extractor_name="LineFit")

    # misc features
    weight_dict = I3DictValueExtractor(
        keys=["I3MCWeightDict"], extractor_name="MCWeightDict"
    )
    filter_mask = I3FilterMapExtractor(
        key="FilterMask", extractor_name="FilterMask"
    )

    # put all extractors in a list
    extractors = [
        combined_extractor,
        SRTInIcePulses,
        SplineMPE,
        LineFit,
        weight_dict,
        filter_mask,
        highest_e_in_ice,
        highest_e_daughter,
        highest_e_in_ice_extended,
        highest_e_daughter_extended,
    ]

    # filters
    filters = [
        NullSplitI3Filter(),
        TableFilter(table_name="SRTInIcePulses"),
        TableFilter(table_name="Homogenized_QTot"),
    ]

    # Create the converter object
    converter: DataConverter = I3ToSQLiteConverter(
        extractors=extractors,
        outdir=outdir,
        num_workers=workers,
        gcd_rescue=gcd_rescue,
        i3_filters=filters,
        max_table_size=max_table_size,
    )
    # run the converter
    converter(inputs)
    # merge files removing the db files after merging to save space.
    if merge is True:
        if remove is True:
            converter.merge_files(remove_originals=True)
        else:
            converter.merge_files(remove_originals=False)


if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Convert I3 files to an intermediate format.
"""
        )

        parser.add_argument("--merge", type=bool, default=True)
        parser.add_argument("--remove", type=bool, default=True)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--max_table_size", type=int, default=int(25e7))
        parser.add_argument("--padding", type=int, default=250)
        args, unknown = parser.parse_known_args()
        # Run example script
        main(
            merge=args.merge,
            remove=args.remove,
            workers=args.workers,
            max_table_size=args.max_table_size,
            padding=args.padding,
        )
