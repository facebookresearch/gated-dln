from src.config_builder.resolvers import register_new_resolvers


def prepare_for_loading_configs():
    register_new_resolvers()
