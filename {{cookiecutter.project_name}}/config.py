from dynaconf import Dynaconf

CFG = Dynaconf(envvar_prefix="DYNACONF", settings_files=["config/config.yaml"])
