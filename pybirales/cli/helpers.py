def update_config(config, section, key, value):
    if config:
        if section in config:
            config[section][key] = value
            return config
        config[section] = {}
        return update_config(config, section, key, value)
    return update_config({}, section, key, value)
