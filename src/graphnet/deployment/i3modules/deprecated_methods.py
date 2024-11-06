"""Contains deprecated methods."""

from typing import Union, Sequence

# from graphnet.deployment.icecube import I3Deployer, I3InferenceModule
from ..icecube.i3deployer import I3Deployer
from ..icecube.inference_module import I3InferenceModule


class GraphNeTI3Deployer(I3Deployer):
    """Class has been renamed to `I3Deployer`.

    Please use `I3Deployer` instead.
    """

    def __init__(
        self,
        graphnet_modules: Union[
            I3InferenceModule, Sequence[I3InferenceModule]
        ],
        gcd_file: str,
        n_workers: int = 1,
    ) -> None:
        """Initialize `GraphNeTI3Deployer`.

        Will apply `DeploymentModules` to files in the order in which they
        appear in `modules`. Each module is run independently.

        Args:
            graphnet_modules: List of `DeploymentModules`.
                              Order of appearence in the list determines order
                              of deployment.
            gcd_file: path to gcd file.
            n_workers: Number of workers. The deployer will divide the number
                       of input files across workers. Defaults to 1.
        """
        super().__init__(
            modules=graphnet_modules, n_workers=n_workers, gcd_file=gcd_file
        )
        self.warning(
            f"{self.__class__} will be deprecated in GraphNeT 2.0"
            " Please use `I3Deployer` instead. "
            " E.g.: `from graphnet.deployment.icecube import I3Deployer`"
        )
