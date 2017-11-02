from SetupBuilder import Builder


build = Builder()

class Parser:

    def parse_to_ini(self, filename):

        setupData = (
        '[General]\n'
        'version = 2.6.0\n\n'
        
        '[simulator]\n'
        'double_precision = true\n\n'
        
        '[sky]\n'
        'oskar_sky_model/file = ' + str(build.sky.Dir) + '\n\n'
        
        '[observation]\n'
        'num_channels = ' + str(build.observation.Chan) + '\n'
        'start_frequency_hz = ' + str(build.observation.StartFreq) + 'e6\n'
        'frequency_inc_hz = ' + str(build.observation.IncreFreq) + 'e6\n'
        'phase_centre_ra_deg = ' + str(build.observation.PCRA) + '\n'
        'phase_centre_dec_deg = ' + str(build.observation.PCDec) + '\n'
        'num_time_steps = ' + str(build.observation.TSteps) + '\n'
        'start_time_utc = ' + str(build.observation.TStart) + '\n'
        'length = ' + str(build.observation.Length) + '\n\n'
        
        '[telescope]\n'
        'longitude_deg = ' + str(build.telescope.Long) + '\n'
        'latitude_deg = ' + str(build.telescope.Lati) + '\n'
        'pol_mode = ' + str(build.telescope.PolM) + '\n'
        'station_type = Aperture_array\n'
        'normalise_beams_at_phase_centre = false\n'
        'aperture_array/element_pattern/enable_numerical = true\n'
        'aperture_array/array_pattern/normalise = false\n'
        'input_directory = ' + str(build.telescope.Dir) + '\n\n'
        
        '[interferometer]\n'
        'oskar_vis_filename = ' + str(build.interferometer.VisF_Dir) + '\n'
        'channel_bandwidth_hz = ' + str(build.interferometer.Bandwith) + 'e6\n'
        'time_average_sec = ' + str(build.interferometer.TAverage) + '\n'
        'ms_filename = ' + str(build.interferometer.MS_Dir) + '\n\n'
        
        '[beam_pattern]\n'
        'coordinate_frame = ' + str(build.beam_pattern.CoordFrame) + '\n'
        'root_path = ' + str(build.beam_pattern.Dir) + '\n'
        'beam_image/size = ' + str(build.beam_pattern.Img_Size) + '\n'
        'beam_image/fov_deg = ' + str(build.beam_pattern.Img_FOV) + '\n'
        'station_outputs/fits_image/amp = true\n\n'
        
        '[image]\n'
        'fov_deg = ' + str(build.image.Img_FOV) + '\n'
        'size = ' + str(build.image.Img_Size) + '\n'
        'image_type = ' + str(build.image.Img_Type) + '\n'
        'time_snapshots = true\n'
        'input_vis_data = ' + str(build.interferometer.VisF_Dir) + '\n'
        'root_path = ' + str(build.image.Dir) + '\n'
        'fits_image = true\n'
        )

        setup_file = open(filename, 'w')
        setup_file.write(setupData)
        setup_file.close()
