from allennlp.common import Params
import logging
logging.basicConfig(level=logging.INFO)


experiment_config: Params = Params.from_file("config.jsonnet")
