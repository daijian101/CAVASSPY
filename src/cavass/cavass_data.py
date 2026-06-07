import struct
from pathlib import Path

import numpy as np


class CAVASS:
    def __init__(self, input_file: str | Path = None, header_only: bool = False):
        # Dimensions
        self.z_dim = 0
        self.y_dim = 0
        self.x_dim = 0
        # Spacings
        self.dx = 1.0
        self.dy = 1.0
        self.dz = 1.0

        # Domain
        self.domain = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        self.axis_labels = b'x\\y\\z'

        # modality (CT, NM, MR, DS, DR, US, OT)
        self.modality = ''

        self.patient_name = ''
        self.patient_id = ''

        self.study_date = ''
        self.study_time = ''

        self.institution = ''

        # Data properties
        self.num_bits = 16
        self.is_signed = True
        self._data = None
        self._data_offset = 0

        # 0 - kilometer, 1 - meter, 2 - cm, 3 - mm, 4 - micron, 5 - sec, 6 - msec, 7 - microsec
        self.measure_unit = 3

        if input_file is not None:
            self.read(input_file, header_only)

    def read(self, input_file: str | Path, header_only: bool = False):
        tags = {}
        with open(input_file, 'rb') as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                group, element, length = struct.unpack('>HHi', header)

                # Get image data position
                if group in (0x7FE0, 0x8001, 0x8021) and element == 0x0010:
                    self._data_offset = f.tell()
                    break

                if element == 0x0000:
                    if length == 4: f.read(4)
                    continue

                value_bytes = f.read(length)
                if group not in tags: tags[group] = {}
                tags[group][element] = value_bytes

            def get_str(g, e):
                return tags.get(g, {}).get(e, b'').decode('ascii').strip().replace('\x00', '')

            def get_shorts(g, e):
                b = tags.get(g, {}).get(e, b'')
                return struct.unpack('>' + 'h' * (len(b) // 2), b) if b else []

            def parse_fstr(g, e):
                s = get_str(g, e)
                return [float(v) for v in s.split('\\')] if s else []

            # Group 0x0008 & 0x0010
            self.study_date = get_str(0x0008, 0x0020)
            self.study_time = get_str(0x0008, 0x0030)
            self.modality = get_str(0x0008, 0x0060)
            self.institution = get_str(0x0008, 0x0080)
            self.patient_name = get_str(0x0010, 0x0010)
            self.patient_id = get_str(0x0010, 0x0020)

            # Group 0x0029 (Scene)
            if 0x0029 in tags:
                s_tags = tags[0x0029]
                self.is_signed = (get_shorts(0x0029, 0x8070)[0] != 0) if 0x8070 in s_tags else True
                self.num_bits = get_shorts(0x0029, 0x8080)[0] if 0x8080 in s_tags else 16

                xy = get_shorts(0x0029, 0x8095)
                if len(xy) >= 2: self.x_dim, self.y_dim = xy[0], xy[1]

                z_val = get_shorts(0x0029, 0x80A0)
                if z_val: self.z_dim = z_val[0]

                sp = parse_fstr(0x0029, 0x80A5)
                if len(sp) >= 2: self.dx, self.dy = sp[0], sp[1]

                locs = parse_fstr(0x0029, 0x80B0)
                if len(locs) >= 2: self.dz = locs[1] - locs[0]

                self.domain = parse_fstr(0x0029, 0x8010) or self.domain
                self.axis_labels = s_tags.get(0x8015, self.axis_labels)

            # Read image
            if not header_only:
                f.seek(self._data_offset)
                if self.num_bits == 1:
                    bytes_per_slice = (self.x_dim * self.y_dim + 7) // 8
                    data = np.zeros((self.z_dim, self.y_dim, self.x_dim), dtype=np.uint8)
                    for i in range(self.z_dim):
                        slice_data = f.read(bytes_per_slice)
                        unpacked = np.unpackbits(np.frombuffer(slice_data, dtype=np.uint8))
                        data[i] = unpacked[:self.x_dim * self.y_dim].reshape((self.y_dim, self.x_dim))
                    self._data = data
                else:
                    dt = {8: np.int8 if self.is_signed else np.uint8,
                          16: '>i2' if self.is_signed else '>u2',
                          32: '>i4' if self.is_signed else '>u4'}.get(self.num_bits, '>i2')
                    raw = f.read(self.z_dim * self.x_dim * self.y_dim * (self.num_bits // 8))
                    data = np.frombuffer(raw, dtype=dt).reshape((self.z_dim, self.y_dim, self.x_dim))
                    self._data = data.copy()

        return self

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        self.z_dim, self.y_dim, self.x_dim = value.shape
        if value.dtype == bool or (value.max() <= 1 and value.min() >= 0):
            self.num_bits, self.is_signed = 1, False
        else:
            self.num_bits = 16
            self.is_signed = np.issubdtype(value.dtype, np.signedinteger)

    @property
    def voxel_spacing(self):
        return self.dz, self.dy, self.dx

    @staticmethod
    def _pack_tag(g, e, p):
        return struct.pack('>HHi', g, e, len(p)) + p

    @staticmethod
    def _fstr(vals):
        return '\\'.join([f'{float(v):e}' for v in vals]).encode('ascii')

    @staticmethod
    def _build_group(group_num, tags_dict):
        payload = b''

        # Write Length to End to Ident group (0x0008)
        if group_num == 0x0008:
            payload += CAVASS._pack_tag(group_num, 0x0001, struct.pack('>i', 0))

        for elem in sorted(tags_dict.keys()):
            val = tags_dict[elem]
            if isinstance(val, bytes) and len(val) % 2 != 0:
                val += b' '
            payload += CAVASS._pack_tag(group_num, elem, val)

        grp_len_tag = struct.pack('>HHiI', group_num, 0x0000, 4, len(payload))
        return grp_len_tag + payload

    @staticmethod
    def _write_empty_group(f, group_num):
        f.write(struct.pack('>HHiI', group_num, 0x0000, 4, 0))

    def save(self, output_file: str | Path):
        if self._data is None:
            raise ValueError('This CAVASS object doesn\'t contain any image data.')

        output_file = Path(output_file) if isinstance(output_file, str) else output_file
        directory = output_file.parent
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            f.write(struct.pack('>HHiI', 0x0000, 0x0000, 4, 12))
            f.write(struct.pack('>HHiI', 0x0000, 0x0001, 4, 0))

            # Ident Group (0x0008)
            ident_tags = {
                0x0010: b'VIEWNIX1.0',
                0x0040: struct.pack('>h', 0)  # 0 = IMAGE0
            }
            if self.study_date: ident_tags[0x0020] = self.study_date.encode('ascii')
            if self.study_time: ident_tags[0x0030] = self.study_time.encode('ascii')
            if self.modality:   ident_tags[0x0060] = self.modality.encode('ascii')
            if self.institution: ident_tags[0x0080] = self.institution.encode('ascii')
            f.write(self._build_group(0x0008, ident_tags))

            self._write_empty_group(f, 0x0009)

            # Patient Group (0x0010)
            if self.patient_name or self.patient_id:
                pa_tags = {}
                if self.patient_name: pa_tags[0x0010] = self.patient_name.encode('ascii')
                if self.patient_id:   pa_tags[0x0020] = self.patient_id.encode('ascii')
                f.write(self._build_group(0x0010, pa_tags))
            else:
                self._write_empty_group(f, 0x0010)

            self._write_empty_group(f, 0x0018)
            self._write_empty_group(f, 0x0020)
            self._write_empty_group(f, 0x0028)

            # Scene Group (0x0029)
            scene_tags = {
                0x8000: struct.pack('>h', 3),
                0x8010: self._fstr(self.domain),
                0x8015: self.axis_labels,
                0x8020: struct.pack('>hhh', 3, 3, 3),
                0x8030: struct.pack('>h', 1),
                0x8035: struct.pack('>h', 0),
                0x8040: self._fstr([float(self._data.min())]),
                0x8050: self._fstr([float(self._data.max())]),
                0x8060: struct.pack('>h', 1),
                0x8070: struct.pack('>h', 1 if self.is_signed else 0),
                0x8080: struct.pack('>h', self.num_bits),
                0x8090: struct.pack('>hh', 0, 0 if self.num_bits == 1 else self.num_bits - 1),
                0x8091: struct.pack('>h', 2),
                0x8092: struct.pack('>h', 1),
                0x8095: struct.pack('>hh', self.x_dim, self.y_dim),
                0x80A0: struct.pack('>h', self.z_dim),
                0x80A5: self._fstr([self.dx, self.dy]),
                0x80B0: self._fstr([i * self.dz for i in range(self.z_dim)])
            }
            f.write(self._build_group(0x0029, scene_tags))

            # Structure & Display
            self._write_empty_group(f, 0x002B)
            self._write_empty_group(f, 0x002D)

            # Image data
            if self.num_bits == 1:
                bytes_per_slice = (self.x_dim * self.y_dim + 7) // 8
                pixel_bytes = bytearray()
                for i in range(self.z_dim):
                    flat_slice = self._data[i].flatten()
                    pad_len = (bytes_per_slice * 8) - len(flat_slice)
                    if pad_len > 0: flat_slice = np.pad(flat_slice, (0, pad_len), 'constant', constant_values=0)
                    pixel_bytes.extend(np.packbits(flat_slice).tobytes())
                pixel_data = bytes(pixel_bytes)
            else:
                dt = (
                    f'>i{self.num_bits // 8}' if self.is_signed else f'>u{self.num_bits // 8}') if self.num_bits > 8 else (
                    'int8' if self.is_signed else 'uint8')
                pixel_data = self._data.astype(dt).tobytes()

            pix_len = len(pixel_data)
            f.write(struct.pack('>HHiI', 0x7FE0, 0x0000, 4, pix_len + 8))
            f.write(struct.pack('>HHi', 0x7FE0, 0x0010, pix_len))
            f.write(pixel_data)

            end_pos = f.tell()

            # Command Group Message Length at offset 20
            f.seek(20, 0)
            f.write(struct.pack('>i', end_pos - 24))

            # Ident Group Message Length at offset 44
            f.seek(44, 0)
            f.write(struct.pack('>i', end_pos - 48))

    def from_template(self, reference_cavass_obj):
        self.dx = reference_cavass_obj.dx
        self.dy = reference_cavass_obj.dy
        self.dz = reference_cavass_obj.dz

        self.domain = reference_cavass_obj.domain
        self.axis_labels = reference_cavass_obj.axis_labels

        self.modality = reference_cavass_obj.modality

        self.patient_name = reference_cavass_obj.patient_name
        self.patient_id = reference_cavass_obj.patient_id

        self.study_date = reference_cavass_obj.study_date
        self.study_time = reference_cavass_obj.study_time

        self.measure_unit = reference_cavass_obj.measure_unit

        return self
