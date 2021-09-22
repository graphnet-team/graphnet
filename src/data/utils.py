from icecube import dataclasses, icetray, dataio

def load_geospatial_data(gcd_path):
    gcd_file = dataio.I3File(gcd_path)
    g_frame = gcd_file.pop_frame(icetray.I3Frame.Geometry)
    om_geom_dict = g_frame["I3Geometry"].omgeo
    calibration = gcd_file.pop_frame(icetray.I3Frame.Calibration)["I3Calibration"]
    return om_geom_dict, calibration  

def build_retroreco_extraction(is_mc):
    ''' Builds a default RetroReco extraction dictionary. Contains associated quantities, such as BDT classifiers.

        is_mc               : boolean, if True the estimated weight of the simulated event is extracted too.

        RETURNS
        retro_extraction    : dictionary where each field contains a string that is evaluated later using eval()

    '''
    retro_extraction = {'azimuth_retro': 'frame["L7_reconstructed_azimuth"].value',
                        'time_retro': 'frame["L7_reconstructed_time"].value',
                        'energy_retro': 'frame["L7_reconstructed_total_energy"].value', 
                        'position_x_retro': 'frame["L7_reconstructed_vertex_x"].value', 
                        'position_y_retro': 'frame["L7_reconstructed_vertex_y"].value',
                        'position_z_retro': 'frame["L7_reconstructed_vertex_z"].value',
                        'zenith_retro': 'frame["L7_reconstructed_zenith"].value',
                        'azimuth_sigma': 'frame["L7_retro_crs_prefit__azimuth_sigma_tot"].value',
                        'position_x_sigma': 'frame["L7_retro_crs_prefit__x_sigma_tot"].value',
                        'position_y_sigma': 'frame["L7_retro_crs_prefit__y_sigma_tot"].value',
                        'position_z_sigma': 'frame["L7_retro_crs_prefit__z_sigma_tot"].value',
                        'time_sigma': 'frame["L7_retro_crs_prefit__time_sigma_tot"].value',
                        'zenith_sigma': 'frame["L7_retro_crs_prefit__zenith_sigma_tot"].value',
                        'energy_sigma': 'frame["L7_retro_crs_prefit__energy_sigma_tot"].value',
                        'cascade_energy_retro': 'frame["L7_reconstructed_cascade_energy"].value',
                        'track_energy_retro': 'frame["L7_reconstructed_track_energy"].value',
                        'track_length_retro': 'frame["L7_reconstructed_track_length"].value',
                        'L7_MuonClassifier_FullSky_ProbNu': 'frame["L7_MuonClassifier_FullSky_ProbNu"].value',
                        'L4_MuonClassifier_Data_ProbNu': 'frame["L4_MuonClassifier_Data_ProbNu"].value',
                        'L4_NoiseClassifier_ProbNu': 'frame["L4_NoiseClassifier_ProbNu"].value',
                        'L7_PIDClassifier_FullSky_ProbTrack': 'frame["L7_PIDClassifier_FullSky_ProbTrack"].value'}

    if is_mc:
        retro_extraction['osc_weight'] = 'frame["I3MCWeightDict"]["weight"]'    
    return retro_extraction

def build_standard_extraction():
    ''' Builds the standard truth extraction.

        RETURNS
        standard_truths    : dictionary where each field contains a string that is evaluated later using eval()
    '''
    standard_truths = {'energy': 'MCInIcePrimary.energy',
            'position_x': 'MCInIcePrimary.pos.x', 
            'position_y': 'MCInIcePrimary.pos.y', 
            'position_z': 'MCInIcePrimary.pos.z',
            'azimuth': 'MCInIcePrimary.dir.azimuth',
            'zenith': 'MCInIcePrimary.dir.zenith',
            'pid': 'MCInIcePrimary.pdg_encoding',
            'event_time': 'event_time',
            'sim_type': 'sim_type',
            'interaction_type': 'interaction_type',
            'elasticity': 'elasticity',
            'RunID': 'RunID',
            'SubrunID': 'SubrunID',
            'EventID': 'EventID',
            'SubEventID': 'SubEventID'}
    return standard_truths

def build_blank_extraction(padding_value = -1):
    ''' Builds the standard blank truth extraction.

        padding_value      : integer or float, the value used for padding 

        RETURNS
        blank_extraction    : dictionary where each field contains a string that is evaluated later using eval()
    '''
    ## Please note that if the simulation type is pure noise or real that these values will be appended to the truth table
    blank_extraction = {'energy': str(padding_value),
            'position_x': str(padding_value), 
            'position_y': str(padding_value), 
            'position_z': str(padding_value),
            'azimuth': str(padding_value),
            'zenith': str(padding_value),
            'pid': str(padding_value),
            'event_time': 'event_time',
            'sim_type': 'sim_type',
            'interaction_type': str(padding_value),
            'elasticity': str(padding_value),
            'RunID': 'RunID',
            'SubrunID': 'SubrunID',
            'EventID': 'EventID',
            'SubEventID': 'SubEventID'}
    return blank_extraction



class I3Extractor:
    def __call__(self, frame, mode, pulsemap, gcd_dict, calibration, input_file, custom_truth = None):
        ''' Extract the truth and pulse map information from frame 
            frame       : I3 Physics Frame 
            mode        : string 
        '''
        if mode == 'oscNext' or mode == 'NuGen':
            truth, pulsemap, retro = self._oscnext_extractor(frame, pulsemap, gcd_dict, calibration, input_file, custom_truth)
            return truth, pulsemap, retro
        elif mode == 'inference':
            pulsemap =  self._extract_features(frame, pulsemap, gcd_dict,calibration)
            return None, pulsemap
        else:
            print('ERROR: invalid mode got : %s'%str(mode))
            return None, None
        
    def _oscnext_extractor(self,frame, pulsemap, gcd_dict, calibration, input_file, custom_truth = None):
        ''' Extract NuGen or oscNext simulation events

            frame       : i3 physics frame
            pulsemap    : string, determining the pulsemap for extraction
            gcd_dict    : dictionary, containing the geospatial information from the gcd-file
            calibration : the i3 physics frame calibration
            input_file  : string, path to the i3 file for extraction
            custom_truth: dictionary, where each field contains a string for evaluation using eval() (check extract_standard_truths() for reference)

            RETURNS
            truths      : dictionary containing the truth information of the frame
            features    : dictionary containing xyz, charge, time, relative dom efficiency, pmt area and pulse width
        '''
        features = self._extract_features(frame, pulsemap, gcd_dict,calibration)
        truths   = self._extract_truth(frame, input_file, custom_truth)
        retros   = self._extract_retro(frame)
        return truths, features, retros
    
    def _extract_features(self,frame, pulsemap, gcd_dict,calibration):
        ''' Extracts the chosen pulsemap from the chosen frame object. Handles known pulsemap errors.

            frame       : i3 physics frame
            pulsemap    : string, eg. 'SRTInIcePulses'
            gcd_dict    : dictionary containing the geospatial information from GCD-file
            calibration : i3 file calibration

            RETURNS
            pulsemap    : dictionary containing charge, time, xyz, pmt area and relative dom efficiency
        '''
        charge = []
        time   = []
        width  = []
        area   = []
        rqe    = []
        x       = []
        y       = []
        z       = []
        if pulsemap in frame.keys():
            om_keys,data = self._get_om_keys(frame, pulsemap, calibration)
            for om_key in om_keys:
                pulses = data[om_key]
                for pulse in pulses:
                    charge.append(pulse.charge)
                    time.append(pulse.time) 
                    width.append(pulse.width)
                    area.append(gcd_dict[om_key].area)  
                    rqe.append(frame["I3Calibration"].dom_cal[om_key].relative_dom_eff)
                    x.append(gcd_dict[om_key].position.x)
                    y.append(gcd_dict[om_key].position.y)
                    z.append(gcd_dict[om_key].position.z)
            
        features = {'charge': charge, 
                    'dom_time': time, 
                    'dom_x': x, 
                    'dom_y': y, 
                    'dom_z': z,
                    'width' : width,
                    'pmt_area': area, 
                    'rde': rqe}
        return features
    def _extract_truth(self,frame, input_file, extract_these_truths = None):
        '''Extracts the truth information in the frame. 
        If a custom extraction scheme is not provided, it defaults to a standard extraction.

        frame                : i3 physics frame
        input_file           : string, path to i3 file 
        extract_these_truths : dictionary, containing strings in fields for evaluation using eval() (see standard_truth_extraction() for reference)
        
        RETURNS

        truth                : dictionary, containing strings in fields for evaluation using eval()
        '''
        if extract_these_truths == None:
            extract_these_truths = build_standard_extraction()
        mc = self._is_montecarlo(frame)
        sim_type = self._find_data_type(mc,input_file)
        event_time =  frame['I3EventHeader'].start_time.utc_daq_time
        RunID, SubrunID, EventID, SubEventID = self._extract_event_ids(frame)
        if mc:
            MCInIcePrimary, interaction_type, elasticity = self._case_handle_this(frame, sim_type)
            if MCInIcePrimary != None:
                ## is not noise
                truth = {}
                for truth_variable in extract_these_truths.keys():
                    truth[truth_variable] = eval(extract_these_truths[truth_variable])
            else:
                print('Could Not Find Primary Particle')
        else:
            ## is real data or noise   
            blank_extraction = build_blank_extraction()
            truth = {}
            for truth_variable in blank_extraction.keys():
                truth[truth_variable] = eval(blank_extraction[truth_variable])
        return truth
    def _extract_retro(self,frame):
        contains_retro = self._contains_retroreco(frame)
        contains_classifier = self._contains_classifiers(frame)
        is_mc = self._is_montecarlo(frame)
        retro = {}
        if contains_retro or contains_classifier:
            retro_extraction = build_retroreco_extraction(is_mc)
            for retro_variable in retro_extraction.keys():
                retro[retro_variable] = eval(self.evaluate_expression(retro_extraction[retro_variable],frame)) 
        return retro
    def _get_om_keys(self,frame, pulsemap, calibration):
        data    = frame[pulsemap]
        try:
            om_keys = data.keys()
        except:
            try:
                if "I3Calibration" in frame.keys():
                    data = frame[pulsemap].apply(frame)
                    om_keys = data.keys()
                else:
                    frame["I3Calibration"] = calibration 
                    data = frame[pulsemap].apply(frame)
                    om_keys = data.keys()
            except:
                data = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,pulsemap)
                om_keys = data.keys()
        return om_keys, data
    def _contains_retroreco(self,frame):
        try:
            frame['L7_reconstructed_zenith']
            return True
        except:
            return False
    def evaluate_expression(self,expression,frame, padding_value = -1):
        try:
            eval(expression)
            out = expression
        except:
            out = str(padding_value)
        return out

    def _is_montecarlo(self,frame):
        ''' Checks if the frame is a simulated event

            frame       : i3 physics frame

            RETURNS
            mc          : boolean
        '''
        mc = True
        try:
            frame['MCInIcePrimary']
        except:
            try:
                frame['I3MCTree']
            except:
                mc = False
        return mc

    def _contains_classifiers(self,frame):
        try:
            frame["L4_MuonClassifier_Data_ProbNu"].value
            return True
        except:
            return False

    def _find_data_type(self,mc, input_file):
        ''' A series of checks that determines if the i3-files are real data measurements or MC.
            In case of MC, it determines which type.

            mc         : boolean
            input_file : string, describing the path to the i3-file currently being extracted.

            RETURNS
            sim_type  : string
        '''
        if mc == False:
            sim_type = 'data'
        else:
            sim_type = 'NuGen'
        if 'muon' in input_file:
            sim_type = 'muongun'
        if 'corsika' in input_file:
            sim_type = 'corsika'
        if 'genie' in input_file:
            sim_type = 'genie'
        if 'noise' in input_file:
            sim_type = 'noise'
        if sim_type == 'lol':
            print('SIM TYPE NOT FOUND!')
        return sim_type

    def _case_handle_this(self,frame, sim_type, padding_value = -1):
        ''' A case handler that does two things
            1) Catches issues related to determining the primary MC particle.
            2) Error handles cases where interaction type and elasticity doesnt exist

            frame           : i3 physics frame
            sim_type        : string, determining the simulation type
            padding_value   : int or float, the value used for padding

            RETURNS
            McInIcePrimary  : the primary particle
            interaction_type: integer, either 1 (charged current), 2 (neutral current), 0 (neither)
            elasticity      : float in ]0,1[ 
        '''
        if sim_type != 'noise':
            try:
                MCInIcePrimary = frame['MCInIcePrimary']
            except:
                MCInIcePrimary = frame['I3MCTree'][0]
        else:
            MCInIcePrimary = None
        try:
            interaction_type =  frame["I3MCWeightDict"]["InteractionType"]
        except:
            interaction_type = padding_value
        try:
            elasticity = frame['I3GENIEResultDict']['y']
        except:
            elasticity = padding_value
        return MCInIcePrimary, interaction_type, elasticity

    def _extract_event_ids(self,frame):
        ''' Extracts relevant information contained in the event header. Usefull for backtracking to original i3-files.

            frame       : i3 physics frame

            RETURNS

            RunID       : integer, denoting the ID of the run (eg. 120000)
            SubrunID    : integer, denoting the ID of the subrun (e.g. 502)
            EventID     : integer, denoting the ID of the event (eg. 5)   #NOTICE: THIS IS NOT A UNIQUE NUMBER BETWEEN I3 FILES
            SubEventID  : integer, denoting the ID of the subevent if the original event is split into multiple (e.g. 2)
        '''
        RunID       = frame['I3EventHeader'].run_id
        SubrunID    = frame['I3EventHeader'].sub_run_id
        EventID     = frame['I3EventHeader'].event_id
        SubEventID  = frame['I3EventHeader'].sub_event_id
        return RunID, SubrunID, EventID, SubEventID


