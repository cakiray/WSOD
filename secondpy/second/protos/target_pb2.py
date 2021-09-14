# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/target.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from secondpy.protos import anchors_pb2 as second_dot_protos_dot_anchors__pb2
from secondpy.protos import similarity_pb2 as second_dot_protos_dot_similarity__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/target.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1asecond/protos/target.proto\x12\rsecond.protos\x1a\x1bsecond/protos/anchors.proto\x1a\x1esecond/protos/similarity.proto\"\xb9\x04\n\x0c\x43lassSetting\x12G\n\x17\x61nchor_generator_stride\x18\x01 \x01(\x0b\x32$.second.protos.AnchorGeneratorStrideH\x00\x12\x45\n\x16\x61nchor_generator_range\x18\x02 \x01(\x0b\x32#.second.protos.AnchorGeneratorRangeH\x00\x12,\n\tno_anchor\x18\x03 \x01(\x0b\x32\x17.second.protos.NoAnchorH\x00\x12O\n\x1cregion_similarity_calculator\x18\x04 \x01(\x0b\x32).second.protos.RegionSimilarityCalculator\x12\x1b\n\x13use_multi_class_nms\x18\x05 \x01(\x08\x12\x16\n\x0euse_rotate_nms\x18\x06 \x01(\x08\x12\x18\n\x10nms_pre_max_size\x18\x07 \x01(\x05\x12\x19\n\x11nms_post_max_size\x18\x08 \x01(\x05\x12\x1b\n\x13nms_score_threshold\x18\t \x01(\x02\x12\x19\n\x11nms_iou_threshold\x18\n \x01(\x02\x12\x19\n\x11matched_threshold\x18\x0b \x01(\x02\x12\x1b\n\x13unmatched_threshold\x18\x0c \x01(\x02\x12\x12\n\nclass_name\x18\r \x01(\t\x12\x18\n\x10\x66\x65\x61ture_map_size\x18\x0e \x03(\x03\x42\x12\n\x10\x61nchor_generator\"\x87\x02\n\x0eTargetAssigner\x12\x33\n\x0e\x63lass_settings\x18\x01 \x03(\x0b\x32\x1b.second.protos.ClassSetting\x12 \n\x18sample_positive_fraction\x18\x02 \x01(\x02\x12\x13\n\x0bsample_size\x18\x03 \x01(\r\x12\x18\n\x10\x61ssign_per_class\x18\x04 \x01(\x08\x12\x19\n\x11nms_pre_max_sizes\x18\x05 \x03(\x03\x12\x1a\n\x12nms_post_max_sizes\x18\x06 \x03(\x03\x12\x1c\n\x14nms_score_thresholds\x18\x07 \x03(\x03\x12\x1a\n\x12nms_iou_thresholds\x18\x08 \x03(\x03\x62\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_anchors__pb2.DESCRIPTOR,second_dot_protos_dot_similarity__pb2.DESCRIPTOR,])




_CLASSSETTING = _descriptor.Descriptor(
  name='ClassSetting',
  full_name='second.protos.ClassSetting',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='anchor_generator_stride', full_name='second.protos.ClassSetting.anchor_generator_stride', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_generator_range', full_name='second.protos.ClassSetting.anchor_generator_range', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='no_anchor', full_name='second.protos.ClassSetting.no_anchor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='region_similarity_calculator', full_name='second.protos.ClassSetting.region_similarity_calculator', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_multi_class_nms', full_name='second.protos.ClassSetting.use_multi_class_nms', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_rotate_nms', full_name='second.protos.ClassSetting.use_rotate_nms', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_pre_max_size', full_name='second.protos.ClassSetting.nms_pre_max_size', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_post_max_size', full_name='second.protos.ClassSetting.nms_post_max_size', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_score_threshold', full_name='second.protos.ClassSetting.nms_score_threshold', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_iou_threshold', full_name='second.protos.ClassSetting.nms_iou_threshold', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='matched_threshold', full_name='second.protos.ClassSetting.matched_threshold', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unmatched_threshold', full_name='second.protos.ClassSetting.unmatched_threshold', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_name', full_name='second.protos.ClassSetting.class_name', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_map_size', full_name='second.protos.ClassSetting.feature_map_size', index=13,
      number=14, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='anchor_generator', full_name='second.protos.ClassSetting.anchor_generator',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=107,
  serialized_end=676,
)


_TARGETASSIGNER = _descriptor.Descriptor(
  name='TargetAssigner',
  full_name='second.protos.TargetAssigner',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_settings', full_name='second.protos.TargetAssigner.class_settings', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_positive_fraction', full_name='second.protos.TargetAssigner.sample_positive_fraction', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_size', full_name='second.protos.TargetAssigner.sample_size', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='assign_per_class', full_name='second.protos.TargetAssigner.assign_per_class', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_pre_max_sizes', full_name='second.protos.TargetAssigner.nms_pre_max_sizes', index=4,
      number=5, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_post_max_sizes', full_name='second.protos.TargetAssigner.nms_post_max_sizes', index=5,
      number=6, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_score_thresholds', full_name='second.protos.TargetAssigner.nms_score_thresholds', index=6,
      number=7, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_iou_thresholds', full_name='second.protos.TargetAssigner.nms_iou_thresholds', index=7,
      number=8, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=679,
  serialized_end=942,
)

_CLASSSETTING.fields_by_name['anchor_generator_stride'].message_type = second_dot_protos_dot_anchors__pb2._ANCHORGENERATORSTRIDE
_CLASSSETTING.fields_by_name['anchor_generator_range'].message_type = second_dot_protos_dot_anchors__pb2._ANCHORGENERATORRANGE
_CLASSSETTING.fields_by_name['no_anchor'].message_type = second_dot_protos_dot_anchors__pb2._NOANCHOR
_CLASSSETTING.fields_by_name['region_similarity_calculator'].message_type = second_dot_protos_dot_similarity__pb2._REGIONSIMILARITYCALCULATOR
_CLASSSETTING.oneofs_by_name['anchor_generator'].fields.append(
  _CLASSSETTING.fields_by_name['anchor_generator_stride'])
_CLASSSETTING.fields_by_name['anchor_generator_stride'].containing_oneof = _CLASSSETTING.oneofs_by_name['anchor_generator']
_CLASSSETTING.oneofs_by_name['anchor_generator'].fields.append(
  _CLASSSETTING.fields_by_name['anchor_generator_range'])
_CLASSSETTING.fields_by_name['anchor_generator_range'].containing_oneof = _CLASSSETTING.oneofs_by_name['anchor_generator']
_CLASSSETTING.oneofs_by_name['anchor_generator'].fields.append(
  _CLASSSETTING.fields_by_name['no_anchor'])
_CLASSSETTING.fields_by_name['no_anchor'].containing_oneof = _CLASSSETTING.oneofs_by_name['anchor_generator']
_TARGETASSIGNER.fields_by_name['class_settings'].message_type = _CLASSSETTING
DESCRIPTOR.message_types_by_name['ClassSetting'] = _CLASSSETTING
DESCRIPTOR.message_types_by_name['TargetAssigner'] = _TARGETASSIGNER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ClassSetting = _reflection.GeneratedProtocolMessageType('ClassSetting', (_message.Message,), dict(
  DESCRIPTOR = _CLASSSETTING,
  __module__ = 'second.protos.target_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.ClassSetting)
  ))
_sym_db.RegisterMessage(ClassSetting)

TargetAssigner = _reflection.GeneratedProtocolMessageType('TargetAssigner', (_message.Message,), dict(
  DESCRIPTOR = _TARGETASSIGNER,
  __module__ = 'second.protos.target_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.TargetAssigner)
  ))
_sym_db.RegisterMessage(TargetAssigner)


# @@protoc_insertion_point(module_scope)
