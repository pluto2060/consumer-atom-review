"""Validator registry of the standalone consumer-atom-review package: only the validators this policy uses."""
from . import consumer_atom_validators as _consumer_atom_validators

VALIDATORS = dict(_consumer_atom_validators.VALIDATORS)
