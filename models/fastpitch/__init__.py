from .networks import FastPitch, FastPitch2Wave

net_config = {'n_mel_channels': 80,
            #   'n_symbols': 148,
              'n_symbols': 178,
              'padding_idx': 0,
              'symbols_embedding_dim': 384,
              'in_fft_n_layers': 6,
              'in_fft_n_heads': 1,
              'in_fft_d_head': 64,
              'in_fft_conv1d_kernel_size': 3,
              'in_fft_conv1d_filter_size': 1536,
              'in_fft_output_size': 384,
              'in_fft_share_weights': False,
              'p_in_fft_dropout': 0.1,
              'p_in_fft_dropatt': 0.1,
              'p_in_fft_dropemb': 0.0,
              'out_fft_n_layers': 6,
              'out_fft_n_heads': 1,
              'out_fft_d_head': 64,
              'out_fft_conv1d_kernel_size': 3,
              'out_fft_conv1d_filter_size': 1536,
              'out_fft_output_size': 384,
              'out_fft_share_weights': False,
              'p_out_fft_dropout': 0.1,
              'p_out_fft_dropatt': 0.1,
              'p_out_fft_dropemb': 0.0,
              'dur_predictor_kernel_size': 3,
              'dur_predictor_filter_size': 256,
              'p_dur_predictor_dropout': 0.1,
              'dur_predictor_n_layers': 2,
              'pitch_predictor_kernel_size': 3,
              'pitch_predictor_filter_size': 256,
              'p_pitch_predictor_dropout': 0.1,
              'pitch_predictor_n_layers': 2,
              'pitch_embedding_kernel_size': 3,
            #   'n_speakers': 1,
              'n_speakers': 256,  
              'speaker_emb_weight': 1.0,
              'n_emotions': 16,
              'emotion_emb_weight': 1.0,
              'energy_predictor_kernel_size': 3,
              'energy_predictor_filter_size': 256,
              'p_energy_predictor_dropout': 0.1,
              'energy_predictor_n_layers': 2,
              'energy_conditioning': True,
              'energy_embedding_kernel_size': 3,    
              }
