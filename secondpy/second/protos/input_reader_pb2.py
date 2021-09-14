# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/input_reader.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from secondpy.protos import preprocess_pb2 as second_dot_protos_dot_preprocess__pb2
from secondpy.protos import sampler_pb2 as second_dot_protos_dot_sampler__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/input_reader.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n second/protos/input_reader.proto\x12\rsecond.protos\x1a\x1esecond/protos/preprocess.proto\x1a\x1bsecond/protos/sampler.proto\"\xbc\x08\n\x0bInputReader\x12\x12\n\nbatch_size\x18\x01 \x01(\r\x12\x33\n\x07\x64\x61taset\x18\x02 \x01(\x0b\x32\".second.protos.InputReader.Dataset\x12\x39\n\npreprocess\x18\x03 \x01(\x0b\x32%.second.protos.InputReader.Preprocess\x12\x16\n\x0emax_num_epochs\x18\x04 \x01(\r\x12\x15\n\rprefetch_size\x18\x05 \x01(\r\x1aW\n\x07\x44\x61taset\x12\x17\n\x0fkitti_info_path\x18\x01 \x01(\t\x12\x17\n\x0fkitti_root_path\x18\x02 \x01(\t\x12\x1a\n\x12\x64\x61taset_class_name\x18\x03 \x01(\t\x1a\xa0\x06\n\nPreprocess\x12\x16\n\x0eshuffle_points\x18\x01 \x01(\x08\x12\x1c\n\x14max_number_of_voxels\x18\x02 \x01(\r\x12*\n\"groundtruth_localization_noise_std\x18\x03 \x03(\x02\x12*\n\"groundtruth_rotation_uniform_noise\x18\x04 \x03(\x02\x12%\n\x1dglobal_rotation_uniform_noise\x18\x05 \x03(\x02\x12$\n\x1cglobal_scaling_uniform_noise\x18\x06 \x03(\x02\x12\"\n\x1aglobal_translate_noise_std\x18\x07 \x03(\x02\x12\x1f\n\x17remove_unknown_examples\x18\x08 \x01(\x08\x12\x13\n\x0bnum_workers\x18\t \x01(\r\x12\x1d\n\x15\x61nchor_area_threshold\x18\n \x01(\x02\x12\"\n\x1aremove_points_after_sample\x18\x0b \x01(\x08\x12*\n\"groundtruth_points_drop_percentage\x18\x0c \x01(\x02\x12(\n groundtruth_drop_max_keep_points\x18\r \x01(\r\x12\x1a\n\x12remove_environment\x18\x0e \x01(\x08\x12/\n\'global_random_rotation_range_per_object\x18\x0f \x03(\x02\x12\x45\n\x13\x64\x61tabase_prep_steps\x18\x10 \x03(\x0b\x32(.second.protos.DatabasePreprocessingStep\x12\x30\n\x10\x64\x61tabase_sampler\x18\x11 \x01(\x0b\x32\x16.second.protos.Sampler\x12\x14\n\x0cuse_group_id\x18\x12 \x01(\x08\x12\x1f\n\x17min_num_of_points_in_gt\x18\x13 \x01(\x03\x12\x15\n\rrandom_flip_x\x18\x14 \x01(\x08\x12\x15\n\rrandom_flip_y\x18\x15 \x01(\x08\x12\x19\n\x11sample_importance\x18\x16 \x01(\x02\x62\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_preprocess__pb2.DESCRIPTOR,second_dot_protos_dot_sampler__pb2.DESCRIPTOR,])




_INPUTREADER_DATASET = _descriptor.Descriptor(
  name='Dataset',
  full_name='second.protos.InputReader.Dataset',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='kitti_info_path', full_name='second.protos.InputReader.Dataset.kitti_info_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kitti_root_path', full_name='second.protos.InputReader.Dataset.kitti_root_path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset_class_name', full_name='second.protos.InputReader.Dataset.dataset_class_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=307,
  serialized_end=394,
)

_INPUTREADER_PREPROCESS = _descriptor.Descriptor(
  name='Preprocess',
  full_name='second.protos.InputReader.Preprocess',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shuffle_points', full_name='second.protos.InputReader.Preprocess.shuffle_points', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_number_of_voxels', full_name='second.protos.InputReader.Preprocess.max_number_of_voxels', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groundtruth_localization_noise_std', full_name='second.protos.InputReader.Preprocess.groundtruth_localization_noise_std', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groundtruth_rotation_uniform_noise', full_name='second.protos.InputReader.Preprocess.groundtruth_rotation_uniform_noise', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_rotation_uniform_noise', full_name='second.protos.InputReader.Preprocess.global_rotation_uniform_noise', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_scaling_uniform_noise', full_name='second.protos.InputReader.Preprocess.global_scaling_uniform_noise', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_translate_noise_std', full_name='second.protos.InputReader.Preprocess.global_translate_noise_std', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='remove_unknown_examples', full_name='second.protos.InputReader.Preprocess.remove_unknown_examples', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_workers', full_name='second.protos.InputReader.Preprocess.num_workers', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_area_threshold', full_name='second.protos.InputReader.Preprocess.anchor_area_threshold', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='remove_points_after_sample', full_name='second.protos.InputReader.Preprocess.remove_points_after_sample', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groundtruth_points_drop_percentage', full_name='second.protos.InputReader.Preprocess.groundtruth_points_drop_percentage', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groundtruth_drop_max_keep_points', full_name='second.protos.InputReader.Preprocess.groundtruth_drop_max_keep_points', index=12,
      number=13, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='remove_environment', full_name='second.protos.InputReader.Preprocess.remove_environment', index=13,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='global_random_rotation_range_per_object', full_name='second.protos.InputReader.Preprocess.global_random_rotation_range_per_object', index=14,
      number=15, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='database_prep_steps', full_name='second.protos.InputReader.Preprocess.database_prep_steps', index=15,
      number=16, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='database_sampler', full_name='second.protos.InputReader.Preprocess.database_sampler', index=16,
      number=17, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_group_id', full_name='second.protos.InputReader.Preprocess.use_group_id', index=17,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_num_of_points_in_gt', full_name='second.protos.InputReader.Preprocess.min_num_of_points_in_gt', index=18,
      number=19, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_flip_x', full_name='second.protos.InputReader.Preprocess.random_flip_x', index=19,
      number=20, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_flip_y', full_name='second.protos.InputReader.Preprocess.random_flip_y', index=20,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sample_importance', full_name='second.protos.InputReader.Preprocess.sample_importance', index=21,
      number=22, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=397,
  serialized_end=1197,
)

_INPUTREADER = _descriptor.Descriptor(
  name='InputReader',
  full_name='second.protos.InputReader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='second.protos.InputReader.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset', full_name='second.protos.InputReader.dataset', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='preprocess', full_name='second.protos.InputReader.preprocess', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_num_epochs', full_name='second.protos.InputReader.max_num_epochs', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prefetch_size', full_name='second.protos.InputReader.prefetch_size', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INPUTREADER_DATASET, _INPUTREADER_PREPROCESS, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=113,
  serialized_end=1197,
)

_INPUTREADER_DATASET.containing_type = _INPUTREADER
_INPUTREADER_PREPROCESS.fields_by_name['database_prep_steps'].message_type = second_dot_protos_dot_preprocess__pb2._DATABASEPREPROCESSINGSTEP
_INPUTREADER_PREPROCESS.fields_by_name['database_sampler'].message_type = second_dot_protos_dot_sampler__pb2._SAMPLER
_INPUTREADER_PREPROCESS.containing_type = _INPUTREADER
_INPUTREADER.fields_by_name['dataset'].message_type = _INPUTREADER_DATASET
_INPUTREADER.fields_by_name['preprocess'].message_type = _INPUTREADER_PREPROCESS
DESCRIPTOR.message_types_by_name['InputReader'] = _INPUTREADER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InputReader = _reflection.GeneratedProtocolMessageType('InputReader', (_message.Message,), dict(

  Dataset = _reflection.GeneratedProtocolMessageType('Dataset', (_message.Message,), dict(
    DESCRIPTOR = _INPUTREADER_DATASET,
    __module__ = 'second.protos.input_reader_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.InputReader.Dataset)
    ))
  ,

  Preprocess = _reflection.GeneratedProtocolMessageType('Preprocess', (_message.Message,), dict(
    DESCRIPTOR = _INPUTREADER_PREPROCESS,
    __module__ = 'second.protos.input_reader_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.InputReader.Preprocess)
    ))
  ,
  DESCRIPTOR = _INPUTREADER,
  __module__ = 'second.protos.input_reader_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.InputReader)
  ))
_sym_db.RegisterMessage(InputReader)
_sym_db.RegisterMessage(InputReader.Dataset)
_sym_db.RegisterMessage(InputReader.Preprocess)


# @@protoc_insertion_point(module_scope)
