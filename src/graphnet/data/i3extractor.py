from abc import ABC, abstractmethod
from typing import List
import numpy as np
import matplotlib.path as mpath
try:
    from icecube import dataclasses, icetray, dataio , phys_services  # pyright: reportMissingImports=false
except ImportError:
    print("icecube package not available.")

from abc import abstractmethod
from .utils import frame_has_key


class I3Extractor(ABC):
    """Extracts relevant information from physics frames."""

    def __init__(self, name):

        # Member variables
        self._i3_file = None
        self._gcd_file = None
        self._gcd_dict = None
        self._calibration = None
        self._name = name

    def set_files(self, i3_file, gcd_file):
        # @TODO: Is it necessary to set the `i3_file`? It is only used in one
        #        place in `I3TruthExtractor`, and there only in a way that might
        #        be solved another way.
        self._i3_file = i3_file
        self._gcd_file = gcd_file
        self._load_gcd_data()

    def _load_gcd_data(self):
        """Loads the geospatial information contained in the gcd-file."""
        gcd_file = dataio.I3File(self._gcd_file)
        g_frame = gcd_file.pop_frame(icetray.I3Frame.Geometry)
        c_frame = gcd_file.pop_frame(icetray.I3Frame.Calibration)
        self._gcd_dict = g_frame["I3Geometry"].omgeo
        self._calibration = c_frame["I3Calibration"]

    @abstractmethod
    def __call__(self, frame) -> dict:
        """Extracts relevant information from frame."""
        pass

    @property
    def name(self) -> str:
        return self._name


class I3ExtractorCollection(list):
    """Class to manage multiple I3Extractors."""
    def __init__(self, *extractors):
        # Check(s)
        for extractor in extractors:
            assert isinstance(extractor, I3Extractor)

        # Base class constructor
        super().__init__(extractors)

    def set_files(self, i3_file, gcd_file):
        for extractor in self:
            extractor.set_files(i3_file, gcd_file)

    def __call__(self, frame) -> List[dict]:
        return [extractor(frame) for extractor in self]


class I3FeatureExtractor(I3Extractor):
    def __init__(self, pulsemap):
        self._pulsemap = pulsemap
        super().__init__(pulsemap)

    def _get_om_keys_and_pulseseries(self, frame):
        """Gets the indicies for the gcd_dict and the pulse series

        Args:
            frame (i3 physics frame): i3 physics frame

        Returns:
            om_keys (index): the indicies for the gcd_dict
            data    (??)   : the pulse series
        """
        data = frame[self._pulsemap]
        try:
            om_keys = data.keys()
        except:
            try:
                if "I3Calibration" in frame.keys():
                    data = frame[self._pulsemap].apply(frame)
                    om_keys = data.keys()
                else:
                    frame["I3Calibration"] = self._calibration
                    data = frame[self._pulsemap].apply(frame)
                    om_keys = data.keys()
                    del frame["I3Calibration"]  # Avoid adding unneccesary data to frame
            except:
                data = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self._pulsemap)
                om_keys = data.keys()
        return om_keys, data

class I3FeatureExtractorIceCube86(I3FeatureExtractor):

    def __call__(self, frame) -> dict:
        """Extract features to be used as inputs to GNN models."""

        output = {
            'charge': [],
            'dom_time': [],
            'dom_x': [],
            'dom_y': [],
            'dom_z': [],
            'width' : [],
            'pmt_area': [],
            'rde': [],
        }

        try:
            om_keys, data = self._get_om_keys_and_pulseseries(frame)
        except KeyError:
            #print(f"WARN: Pulsemap {self._pulsemap} was not found in frame.")
            return output

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(frame, om_key)

            # Loop over pulses for each OM
            pulses = data[om_key]
            for pulse in pulses:
                output['charge'].append(pulse.charge)
                output['dom_time'].append(pulse.time)
                output['width'].append(pulse.width)
                output['pmt_area'].append(area)
                output['rde'].append(rde)
                output['dom_x'].append(x)
                output['dom_y'].append(y)
                output['dom_z'].append(z)

        return output
    def _get_relative_dom_efficiency(self, frame, om_key):
        if "I3Calibration" in frame:  # Not available for e.g. mDOMs in IceCube Upgrade
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:
                rde = -1
        return rde

class I3FeatureExtractorIceCubeDeepCore(I3FeatureExtractorIceCube86):
    """..."""

class I3FeatureExtractorIceCubeUpgrade(I3FeatureExtractorIceCube86):

    def __call__(self, frame) -> dict:
        """Extract features to be used as inputs to GNN models."""

        output = {
            'string': [],
            'pmt_number': [],
            'dom_number': [],
            'pmt_dir_x': [],
            'pmt_dir_y': [],
            'pmt_dir_z': [],
            'dom_type': [],
        }

        try:
            om_keys, data = self._get_om_keys_and_pulseseries(frame)
        except KeyError:  # Target pulsemap does not exist in `frame`
            return output

        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z
            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # Loop over pulses for each OM
            pulses = data[om_key]
            for _ in pulses:
                output['string'].append(string)
                output['pmt_number'].append(pmt_number)
                output['dom_number'].append(dom_number)
                output['pmt_dir_x'].append(pmt_dir_x)
                output['pmt_dir_y'].append(pmt_dir_y)
                output['pmt_dir_z'].append(pmt_dir_z)
                output['dom_type'].append(dom_type)

        # Add features from IceCube86
        output_icecube86 = super().__call__(frame)
        output.update(output_icecube86)
        return output

class I3PulseNoiseTruthFlagIceCubeUpgrade(I3FeatureExtractorIceCube86):

    def __call__(self, frame) -> dict:
        """Extract features to be used as inputs to GNN models."""

        output = {
            'string': [],
            'pmt_number': [],
            'dom_number': [],
            'pmt_dir_x': [],
            'pmt_dir_y': [],
            'pmt_dir_z': [],
            'dom_type': [],
            'truth_flag': [],
        }

        try:
            om_keys, data = self._get_om_keys_and_pulseseries(frame)
        except KeyError:  # Target pulsemap does not exist in `frame`
            return output
            
        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z
            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # Loop over pulses for each OM
            pulses = data[om_key]
            for truth_flag in pulses:
                output['string'].append(string)
                output['pmt_number'].append(pmt_number)
                output['dom_number'].append(dom_number)
                output['pmt_dir_x'].append(pmt_dir_x)
                output['pmt_dir_y'].append(pmt_dir_y)
                output['pmt_dir_z'].append(pmt_dir_z)
                output['dom_type'].append(dom_type)
                output['truth_flag'].append(truth_flag)

        return output

class I3TruthExtractor(I3Extractor):
    def __init__(self, name="truth", borders=None):
        super().__init__(name)
        if borders == None:
            border_xy = np.array([(-256.1400146484375, -521.0800170898438), (-132.8000030517578, -501.45001220703125), (-9.13000011444092, -481.739990234375), (114.38999938964844, -461.989990234375), (237.77999877929688, -442.4200134277344), (361.0, -422.8299865722656), (405.8299865722656, -306.3800048828125), (443.6000061035156, -194.16000366210938), (500.42999267578125, -58.45000076293945), (544.0700073242188, 55.88999938964844), (576.3699951171875, 170.9199981689453), (505.2699890136719, 257.8800048828125), (429.760009765625, 351.0199890136719), (338.44000244140625, 463.7200012207031), (224.5800018310547, 432.3500061035156), (101.04000091552734, 412.7900085449219), (22.11000061035156, 509.5), (-101.05999755859375, 490.2200012207031), (-224.08999633789062, 470.8599853515625), (-347.8800048828125, 451.5199890136719), (-392.3800048828125, 334.239990234375), (-437.0400085449219, 217.8000030517578), (-481.6000061035156, 101.38999938964844), (-526.6300048828125, -15.60000038146973), (-570.9000244140625, -125.13999938964844), (-492.42999267578125, -230.16000366210938), (-413.4599914550781, -327.2699890136719), (-334.79998779296875, -424.5)])
            border_z = np.array([-512.82,524.56])
            self._borders = [border_xy,border_z]
        else:
            self._borders = borders

    def __call__(self, frame, padding_value=-1) -> dict:
        """Extracts truth features."""
        is_mc = frame_is_montecarlo(frame)
        is_noise = frame_is_noise(frame)
        sim_type = find_data_type(is_mc, self._i3_file)

        output = {
            'energy': padding_value,
            'position_x': padding_value,
            'position_y': padding_value,
            'position_z': padding_value,
            'azimuth': padding_value,
            'zenith': padding_value,
            'pid': padding_value,
            'event_time': frame['I3EventHeader'].start_time.utc_daq_time,
            'sim_type': sim_type,
            'interaction_type': padding_value,
            'elasticity': padding_value,
            'RunID': frame['I3EventHeader'].run_id,
            'SubrunID': frame['I3EventHeader'].sub_run_id,
            'EventID': frame['I3EventHeader'].event_id,
            'SubEventID': frame['I3EventHeader'].sub_event_id,
            'dbang_decay_length': self.__extract_dbang_decay_length__(frame, padding_value),
            'track_length': padding_value,
            'stopped_muon': padding_value,
        }

        if is_mc == True and is_noise == False:
            MCInIcePrimary, interaction_type, elasticity = get_primary_particle_interaction_type_and_elasticity(frame, sim_type)
            output.update({
                'energy': MCInIcePrimary.energy,
                'position_x': MCInIcePrimary.pos.x,
                'position_y': MCInIcePrimary.pos.y,
                'position_z': MCInIcePrimary.pos.z,
                'azimuth': MCInIcePrimary.dir.azimuth,
                'zenith': MCInIcePrimary.dir.zenith,
                'pid': MCInIcePrimary.pdg_encoding,
                'interaction_type': interaction_type,
                'elasticity': elasticity,
            })
            if abs(output['pid'])==13:
                output.update({
                    'track_length': MCInIcePrimary.length,
                })
                muon_final = muon_stopped(output,self._borders)
                output.update({
                    'position_x': muon_final['x'], #position_xyz has no meaning for muons. These will now be updated to muon final position, given track length/azimuth/zenith
                    'position_y': muon_final['y'],
                    'position_z': muon_final['z'],
                    'stopped_muon': muon_final['stopped'],
                })                

        return output

    def __extract_dbang_decay_length__(self,frame, padding_value):
        mctree = frame['I3MCTree']
        try:
            p_true = mctree.primaries[0]
            p_daughters = mctree.get_daughters(p_true)        
            if (len(p_daughters) == 2):
                for p_daughter in p_daughters:
                    if p_daughter.type == dataclasses.I3Particle.Hadrons:
                        casc_0_true = p_daughter
                    else:
                        hnl_true = p_daughter
                hnl_daughters = mctree.get_daughters(hnl_true)
            else:
                decay_length  =  padding_value
                hnl_daughters = []

            if (len(hnl_daughters) > 0):    
                for count_hnl_daughters, hnl_daughter in enumerate(hnl_daughters):
                    if not count_hnl_daughters:
                        casc_1_true = hnl_daughter
                    else:
                        assert(casc_1_true.pos == hnl_daughter.pos)
                        casc_1_true.energy = casc_1_true.energy + hnl_daughter.energy
                decay_length = phys_services.I3Calculator.distance(casc_0_true,casc_1_true)/icetray.I3Units.m
                
            else:
                decay_length = padding_value
            return decay_length
        except:
            return padding_value



class I3RetroExtractor(I3Extractor):

    def __init__(self, name="retro"):
        super().__init__(name)

    def __call__(self, frame) -> dict:
        """Extracts RETRO reco. and associated quantities if available."""
        output = {}

        if frame_contains_retro(frame):
            output.update({
                'azimuth_retro': frame["L7_reconstructed_azimuth"].value,
                'time_retro': frame["L7_reconstructed_time"].value,
                'energy_retro': frame["L7_reconstructed_total_energy"].value,
                'position_x_retro': frame["L7_reconstructed_vertex_x"].value,
                'position_y_retro': frame["L7_reconstructed_vertex_y"].value,
                'position_z_retro': frame["L7_reconstructed_vertex_z"].value,
                'zenith_retro': frame["L7_reconstructed_zenith"].value,
                'azimuth_sigma': frame["L7_retro_crs_prefit__azimuth_sigma_tot"].value,
                'position_x_sigma': frame["L7_retro_crs_prefit__x_sigma_tot"].value,
                'position_y_sigma': frame["L7_retro_crs_prefit__y_sigma_tot"].value,
                'position_z_sigma': frame["L7_retro_crs_prefit__z_sigma_tot"].value,
                'time_sigma': frame["L7_retro_crs_prefit__time_sigma_tot"].value,
                'zenith_sigma': frame["L7_retro_crs_prefit__zenith_sigma_tot"].value,
                'energy_sigma': frame["L7_retro_crs_prefit__energy_sigma_tot"].value,
                'cascade_energy_retro': frame["L7_reconstructed_cascade_energy"].value,
                'track_energy_retro': frame["L7_reconstructed_track_energy"].value,
                'track_length_retro': frame["L7_reconstructed_track_length"].value,
            })

        if frame_contains_classifiers(frame):
            classifiers = ['L7_MuonClassifier_FullSky_ProbNu','L4_MuonClassifier_Data_ProbNu','L4_NoiseClassifier_ProbNu','L7_PIDClassifier_FullSky_ProbTrack']
            for classifier in classifiers:
                if frame_has_key(frame, classifier):
                    output.update({classifier : frame[classifier].value})
            #output.update({
            #    'L7_MuonClassifier_FullSky_ProbNu': frame["L7_MuonClassifier_FullSky_ProbNu"].value,
            #    'L4_MuonClassifier_Data_ProbNu': frame["L4_MuonClassifier_Data_ProbNu"].value,
            #    'L4_NoiseClassifier_ProbNu': frame["L4_NoiseClassifier_ProbNu"].value,
            #    'L7_PIDClassifier_FullSky_ProbTrack': frame["L7_PIDClassifier_FullSky_ProbTrack"].value,
            #})

        if frame_is_montecarlo(frame):
            if frame_contains_retro(frame):
                output.update({
                    'osc_weight': frame["I3MCWeightDict"]["weight"],
                })
            else:
                output.update({
                    'osc_weight': -1.,
                })

        return output


# Utilty methods
def frame_contains_retro(frame):
    return frame_has_key(frame, "L7_reconstructed_zenith")

def frame_contains_classifiers(frame):
    return frame_has_key(frame, "L4_MuonClassifier_Data_ProbNu")

def frame_is_montecarlo(frame):
    return (
        frame_has_key(frame, "MCInIcePrimary") or
        frame_has_key(frame, "I3MCTree")
    )
def frame_is_noise(frame):
    try:
        frame['I3MCTree'][0].energy
        return False
    except:
        try:
            frame['MCInIcePrimary'].energy
            return False
        except:
            return True

def frame_is_lvl7(frame):
    return frame_has_key(frame, "L7_reconstructed_zenith")



def find_data_type(mc, input_file):
    """Determines the data type

    Args:
        mc (boolean): is this montecarlo?
        input_file (str): path to i3 file

    Returns:
        str: the simulation/data type
    """
    # @TODO: Rewrite to make automaticallu infer `mc` from `input_file`?
    if mc == False:
        sim_type = 'data'
    else:
        sim_type = 'NuGen'
    if 'muon' in input_file:
        sim_type = 'muongun'
    if 'corsika' in input_file:
        sim_type = 'corsika'
    if 'genie' in input_file or 'nu' in input_file.lower():
        sim_type = 'genie'
    if 'noise' in input_file:
        sim_type = 'noise'
    if 'L2' in input_file:  ## not robust
        sim_type = 'dbang'
    if sim_type == 'lol':
        print('SIM TYPE NOT FOUND!')
    return sim_type

def get_primary_particle_interaction_type_and_elasticity(frame, sim_type, padding_value=-1):
    """"Returns primary particle, interaction type, and elasticity.

    A case handler that does two things
        1) Catches issues related to determining the primary MC particle.
        2) Error handles cases where interaction type and elasticity doesnt exist

    Args:
        frame (i3 physics frame): ...
        sim_type (string): Simulation type
        padding_value (int | float): The value used for padding.

    Returns
        McInIcePrimary (?): The primary particle
        interaction_type (int): Either 1 (charged current), 2 (neutral current), 0 (neither)
        elasticity (float): In ]0,1[
    """
    if sim_type != 'noise':
        try:
            MCInIcePrimary = frame['MCInIcePrimary']
        except:
            MCInIcePrimary = frame['I3MCTree'][0]
        if MCInIcePrimary.energy != MCInIcePrimary.energy: # This is a nan check. Only happens for some muons where second item in MCTree is primary. Weird!
            MCInIcePrimary = frame['I3MCTree'][1] ## for some strange reason the second entry is identical in all variables and has no nans (always muon)
    else:   
        MCInIcePrimary = None
    try:
        interaction_type = frame["I3MCWeightDict"]["InteractionType"]
    except:
        interaction_type = padding_value

    try:
        elasticity = frame['I3GENIEResultDict']['y']
    except:
        elasticity = padding_value

    return MCInIcePrimary, interaction_type, elasticity



def muon_stopped(truth, borders, horizontal_pad = 100., vertical_pad = 100.):
    '''
    Calculates where a simulated muon stops and if this is inside the detectors fiducial volume. 
    IMPORTANT: The final position of the muon is saved in truth extractor/databases as position_x,position_y and position_z.
               This is analogoues to the neutrinos whose interaction vertex is saved under the same name.

    Args:
        truth (dict) : dictionary of already extracted values
        borders (tuple) : first entry xy outline, second z min/max depth. See I3TruthExtractor for hard-code example.
        horizontal_pad (float) : shrink xy plane further with exclusion zone
        vertical_pad (float) : further shrink detector depth with exclusion height
    
    Returns:
        dictionary (dict) : containing the x,y,z co-ordinates of final muon position and contained boolean (0 or 1)
    '''
    #to do:remove hard-coded border coords and replace with GCD file contents using string no's
    border = mpath.Path(borders[0])

    start_pos = np.array([truth['position_x'],
                          truth['position_y'],
                          truth['position_z']])
                          
    travel_vec = -1*np.array([truth['track_length']*np.cos(truth['azimuth'])*np.sin(truth['zenith']),
                              truth['track_length']*np.sin(truth['azimuth'])*np.sin(truth['zenith']),
                              truth['track_length']*np.cos(truth['zenith'])])
    
    end_pos = start_pos+travel_vec

    stopped_xy = border.contains_point((end_pos[0],end_pos[1]),radius=-horizontal_pad) 
    stopped_z = (end_pos[2] > borders[1][0] + vertical_pad) * (end_pos[2] < borders[1][1] - vertical_pad) 

    return {'x' : end_pos[0], 'y' : end_pos[1], 'z' : end_pos[2], 'stopped' : (stopped_xy * stopped_z) }
