# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os
from typing import Any

import yaml
from yacs.config import CfgNode as _CfgNode

from ..utils.file_io import PathManager

BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:
    1. The :meth:`merge_from_file` method supports the "_BASE_" key,
       which allows the new CfgNode to inherit all the attributes from the
       base configuration file.
    2. Keys that start with "COMPUTED_" are treated as insertion-only
       "computed" attributes. They can be inserted regardless of whether
       the CfgNode is frozen or not.
    3. With "allow_unsafe=True", it supports pyyaml tags that evaluate
       expressions in config. See examples in
       https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    """

    @staticmethod
    def load_yaml_with_base(filename: str, allow_unsafe: bool = False):
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.
        Args:
            filename (str): the file name of the current config. Will be used to
                find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.
        Returns:
            (dict): the loaded yaml
        """
        with PathManager.open(filename, "r") as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with open(filename, "r") as f:
                    cfg = yaml.unsafe_load(f)

        def merge_a_into_b(a, b):
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(
                    map(base_cfg_file.startswith, ["/", "https://", "http://"])
            ):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(
                    os.path.dirname(filename), base_cfg_file
                )
            base_cfg = CfgNode.load_yaml_with_base(
                base_cfg_file, allow_unsafe=allow_unsafe
            )
            del cfg[BASE_KEY]

            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False):
        """
        Merge configs from a given yaml file.
        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = CfgNode.load_yaml_with_base(
            cfg_filename, allow_unsafe=allow_unsafe
        )
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    # Forward the following calls to base, but with a check on the BASE_KEY.
    def merge_from_other_cfg(self, cfg_other):
        """
        Args:
            cfg_other (CfgNode): configs to merge from.
        """
        assert (
                BASE_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list: list):
        """
        Args:
            cfg_list (list): list of configs to merge from.
        """
        keys = set(cfg_list[0::2])
        assert (
                BASE_KEY not in keys
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)

    def __setattr__(self, name: str, val: Any):
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(
                        name, old_val, val
                    )
                )
            self[name] = val
        else:
            super().__setattr__(name, val)


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a fastreid CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()
