def build_model(decoder_config, synth_config, ddsp_mode="hpn"):
    from DDSP.models.ddsp_decoder import DDSP_Decoder
    # print(ddsp_mode)
    if ddsp_mode == "hpn":
        from DDSP.models.decoder.hpn_decoder import HpNdecoder
        from DDSP.models.synths.hpn_synth import HpNsynth
        decoder = HpNdecoder(
                            hidden_size=decoder_config["hidden_size"], 
                            num_layers=decoder_config["num_layers"],
                            input_keys=decoder_config["input_keys"],
                            input_sizes=decoder_config["input_sizes"],
                            output_keys=decoder_config["output_keys"],
                            output_sizes=decoder_config["output_sizes"]
                            )
        synth = HpNsynth(   
                         sample_rate=synth_config["sample_rate"],
                         block_size=synth_config["block_size"],
                         reverb_scale=synth_config["reverb_scale"]
                         )
        
    elif ddsp_mode == "wavetable":
        from DDSP.models.decoder.hpn_decoder import HpNdecoder
        from DDSP.models.synths.wavetable_synth import WTpNsynth
        decoder = HpNdecoder(
                            hidden_size=decoder_config["hidden_size"], 
                            num_layers=decoder_config["num_layers"],
                            input_keys=decoder_config["input_keys"],
                            input_sizes=decoder_config["input_sizes"],
                            output_keys=decoder_config["output_keys"],
                            output_sizes=decoder_config["output_sizes"]
                            )
        synth = WTpNsynth(   
                         sample_rate=synth_config["sample_rate"],
                         block_size=synth_config["block_size"],
                         reverb_scale=synth_config["reverb_scale"]
                         )
        
    elif ddsp_mode == "ddx7":
        from DDSP.models.decoder.ddx7_decoder import FMdecoder
        from DDSP.models.synths.ddx7_synth import FMsynth
        
        decoder = FMdecoder(
                    n_blocks=decoder_config["n_blocks"],
                    hidden_channels=decoder_config["hidden_channels"],
                    output_keys=['ol'],
                    out_channels=decoder_config["out_channels"],
                    kernel_size=decoder_config["kernel_size"],
                    dilation_base=decoder_config["dilation_base"],
                    apply_padding=decoder_config["apply_padding"],
                    deploy_residual=decoder_config["deploy_residual"],
                    input_keys=decoder_config["input_keys"],
                    z_size=None,
                    output_complete_controls=True
                    )
        
        synth = FMsynth(
                    sample_rate=synth_config["sample_rate"],
                    block_size=synth_config["block_size"],
                    max_ol=synth_config["max_ol"],
                    fr=synth_config["fr"],
                    synth_module=synth_config["synth_module"]
                    )
    else:
        raise ValueError(f"Configurazione non corretta.")

    # Return the DDSP_Decoder
    return DDSP_Decoder(decoder=decoder, synth=synth)