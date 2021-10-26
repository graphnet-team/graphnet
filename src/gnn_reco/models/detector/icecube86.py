from gnn_reco.models.detector.detector import Detector
from gnn_reco.data.constants import FEATURES
from gnn_reco.models.detector import Detector

class IceCube86(Detector):
    """`Detector` class for IceCube-86.
    """

    # Implementing abstract class attribute
    features = FEATURES.ICECUBE86
    
    def forward(self, data):
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.
            
        Assuming the following features, in this order (see self._features):
            dom_x
            dom_y
            dom_z
            dom_time
            charge
            rde
            pmt_area

        Args:
            data (Data): Input graph data.
        
        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Convenience variables
        assert data.x.size()[1] == self.nb_inputs, f"Got graph data with incompatible size, {data.x.size()} vs. {self.nb_inputs} expected"
        assert self.nb_inputs == 7

        # Graph-bulding
        data = self._graph_builder(data).clone()
        
        # Preprocessing
        data.x[:,0] /= 100.  # dom_x
        data.x[:,1] /= 100.  # dom_y
        data.x[:,2] += 350.  # dom_z
        data.x[:,2] /= 100.  
        data.x[:,3] /= 1.05e+04  # dom_time
        data.x[:,3] -= 1.
        data.x[:,3] *= 20.
        data.x[:,4] /= 1.  # charge
        data.x[:,5] -= 1.25  # rde
        data.x[:,5] /= 0.25  
        data.x[:,6] /= 0.05  # pmt_area

        return data