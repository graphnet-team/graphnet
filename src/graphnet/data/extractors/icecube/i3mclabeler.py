"""Extract Monte Carlo labels from IceCube data.

Using pre-existing IceCube Monte Carlo labeler.
"""

from typing import Any, Dict
from icecube import icetray
from icecube.sim_services.label_events.mc_labeler import MCLabeler


from .i3extractor import I3Extractor


class MClabelerWrapper(MCLabeler):
    """Wrapper for IceCube Monte Carlo labeler."""

    def __init__(self, context: icetray.I3Context):
        """Construct MClabelerWrapper."""
        # Base class constructor
        super().__init__(context=context)

    def GraphNetDAQ(
        self, frame: "icetray.I3Frame", name: str
    ) -> Dict[str, Any]:
        """Extract MC labels from I3Frame.

        Args:
            frame: I3Frame to extract MC labels from.
            name: Name of the extractor.
        """
        output = {}

        if self._geo is None:
            raise RuntimeError("No geometry information found.")
        classif, n_coinc, bg_mcpe, bg_mcpe_charge = self.classify(frame)

        output.update(
            {
                "classification" + name: classif,
                "coincident_muons" + name: n_coinc,
                "bg_muon_mcpe" + name: bg_mcpe,
                "bg_muon_mcpe_charge" + name: bg_mcpe_charge,
            }
        )

        return output


class I3MCLabelerExtractor(I3Extractor):
    """Labeler for IceCube Monte Carlo data."""

    def __init__(self, context: icetray.I3Context, extractor_name: str):
        """Construct I3MCLabeler.

        Args:
            context: IceCube context.
            extractor_name: Name of the extractor.
        """
        # Member variable(s)
        super().__init__(extractor_name=extractor_name)
        # Initialize MCLabeler
        self.labeler = MClabelerWrapper(context=context)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, Any]:
        """Extract MC labels from I3Frame."""
        output = self.labeler.GraphNetDAQ(frame, self._extractor_name)
        return output
