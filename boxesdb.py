import numpy as np
import sqlite3
import json
import os
import logging


def _dumper(obj):
    try:
        return obj.to_json()
    except:
        if isinstance(obj, np.ndarray):
            return [_dumper(el) for el in obj]
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        else:
            return obj.__dict__


class BaseBoxesDB(object):
    def __init__(self, fname):
        self.con = sqlite3.connect(fname)
        self.con.row_factory = sqlite3.Row
        self.log = logging.getLogger('BaseBoxesDB')
        self.fname = fname
        self.labels_table = 'main'
        self.points_table = 'main'
    
    # --- (internal) convenience methods ---
    @staticmethod
    def _get_data_dump(point):
        data = {}
        for key in ('point', 'mesh_point', 'transformed_point', 'spacing'):
            val = point.get(key, None)
            if val is not None:
                data[key] = val
        data_str = json.dumps(data, default=_dumper, sort_keys=True)
        return data_str

    @staticmethod
    def _get_label_dump(label, label_names):
        data = {}
        for l, name in zip(label, label_names):
            data[name] = l
        data_str = json.dumps(data, default=_dumper, sort_keys=True)
        return data_str
    
    @staticmethod
    def _get_label_from_dump(label_str, label_names):
        label_dict = json.loads(label_str)
        label = []
        for name in label_names:
            label.append(label_dict.get(name))  # want this to fail is name does not exist
        return label

    # --- getting / inserting data ---
    def get_point_id(self, point, insert=True, cursor=None):
        if cursor is None:
            cur = self.con.cursor()
        else:
            cur = cursor
        select_query = 'select point_id from {}.points where side = ? and sample_num = ?'.format(self.points_table)
        insert_query = 'insert into {}.points (side, sample_num, slice_no, data) values (?,?,?,?)'.format(self.points_table)
        point_id = cur.execute(select_query,(point['side'], point['sample_num'])).fetchone()
        if point_id is None and insert:
            data = BaseBoxesDB._get_data_dump(point)
            point_id = cur.execute(insert_query,(point['side'], point['sample_num'], point['slice_no'], data)).lastrowid
        else:
            point_id = point_id[0]
        if cursor is None:
            self.con.commit()
        return point_id

    def get_seed_id(self, point, insert=True, cursor=None):
        if cursor is None:
            cur = self.con.cursor()
        else:
            cur = cursor
        select_query = 'select point_id from {}.seeds where side = ? and area = ? and seed_num = ?'.format(self.points_table)
        insert_query = 'insert into {}.seeds (side, area, seed_num, slice_no, data) values (?,?,?,?,?)'.format(self.points_table)
        seed_id = cur.execute(select_query,(point['side'], point['area'], point['seed_num'])).fetchone()
        if seed_id is None and insert:
            data = BaseBoxesDB._get_data_dump(point)
            seed_id = cur.execute(insert_query,(point['side'], point['area'], point['seed_num'], point['slice_no'], data)).lastrowid
        else:
            seed_id = seed_id[0]
        if cursor is None:
            self.con.commit()
        return seed_id

    def insert_label(self, id1, id2, label, label_names):
        query = 'insert into {}.labels (point1, point2, label) values (?,?,?)'.format(self.labels_table)
        with self.con as con:
            label_str = BaseBoxesDB._get_label_dump(label, label_names)
            try:
                con.execute(query, (id1, id2, label_str))
            except sqlite3.IntegrityError:
                self.log.info('Label for {} and {} already exists. Not inserting!'.format(id1, id2))

    def get_label(self, id1, id2, label_names):
        query = 'select label from {}.labels where point1 = ? and point2 = ?'.format(self.labels_table)
        with self.con as con:
            label_str = con.execute(query,(id1, id2)).fetchone()['label']
        return BaseBoxesDB._get_label_from_dump(label_str, label_names)

    def get_all_seeds(self, side, with_label=False):
        query = 'select distinct point_id, side, area, seed_num, slice_no, data from {}.seeds '.format(self.points_table)
        if with_label:
            query += 'inner join {}.labels on point_id = point1 '.format(self.labels_table)
        query += 'where side = ? order by area asc, seed_num asc'
        seeds = []
        ids = []
        with self.con as con:
            for row in con.execute(query,(str(side))):
                seed = {'side': row['side'], 'area': row['area'], 'seed_num': row['seed_num'], 'slice_no': row['slice_no']}
                seed.update(json.loads(row['data']))
                seeds.append(seed)
                ids.append(row['point_id'])
        self.log.info('Got all seeds with label {}. Len seeds: {}'.format(with_label, len(seeds)))
        return seeds, ids

    def get_all_points(self, side):
        query = 'select point_id, side, sample_num, slice_no, data from {}.points where side = ? order by sample_num asc'.format(self.points_table)
        points = []
        ids = []
        with self.con as con:
            for row in con.execute(query,(str(side))):
                point = {'side': row['side'], 'sample_num': row['sample_num'], 'slice_no': row['slice_no']}
                point.update(json.loads(row['data']))
                points.append(point)
                ids.append(row['point_id'])
        return points, ids

    # --- database schema creation ---
    def create_points_table(self):
        with self.con as con:
            con.execute('create table if not exists points (point_id integer primary key, side integer, sample_num integer, slice_no text, data text, unique(side, sample_num))')

    def create_labels_table(self, with_foreign_key=True, point1_table='points', point2_table='points'):
        script = 'create table if not exists labels (point1 integer, point2 integer, label text, unique(point1, point2)'
        if with_foreign_key:
            script += ', foreign key(point1) references '+point1_table+'(point_id), foreign key(point2) references '+point2_table+'(point_id))'
        else:
            script += ')'
        with self.con as con:
            con.executescript(script)

    def create_seeds_table(self):
        with self.con as con:
            con.execute('create table if not exists seeds (point_id integer primary key, side integer, area text, seed_num integer, slice_no text, data text, unique(side, area, seed_num))')


class BoxesDB(BaseBoxesDB):
    def __init__(self, fname):
        super(BoxesDB, self).__init__(fname)
        self.create_points_table()
        self.create_labels_table()

    # --- getting / inserting data ---
    def insert_pair(self, point1, point2, label, label_names):
        num1 = self.get_point_id(point1, insert=True)
        num2 = self.get_point_id(point2, insert=True)
        self.insert_label(num1, num2, label, label_names)

    def batch_insert_pair(self, points, label_names):
        # only works if points and labels are not yet present in the db
        cur = self.con.cursor()
        points1 = [(pt[0]['side'], pt[0]['sample_num'], pt[0]['slice_no'], BaseBoxesDB._get_data_dump(pt[0])) for pt in points]
        ids1 = []
        for pt in points1:
            ids1.append(cur.execute('insert into points (side, sample_num, slice_no, data) values (?,?,?,?)', pt).lastrowid)
        points2 = [(pt[1]['side'], pt[1]['sample_num'], pt[1]['slice_no'], BaseBoxesDB._get_data_dump(pt[1])) for pt in points]
        ids2 = []
        for pt in points2:
            ids2.append(cur.execute('insert into points (side, sample_num, slice_no, data) values (?,?,?,?)', pt).lastrowid)
        labels = [(id1, id2, BaseBoxesDB._get_label_dump(pt[2], label_names)) for id1, id2, pt in zip(ids1, ids2, points)]
        cur.executemany('insert into labels (point1, point2, label) values (?,?,?)', labels)
        self.con.commit()

    def batch_insert_pair_existing(self, points1, points2, labels, label_names, insert=True):
        # only works if labels are not yet present in the db, but points might be existing
        cur = self.con.cursor()
        nums1 = [self.get_point_id(pt, insert=insert, cursor=cur) for pt in points1]
        nums2 = [self.get_point_id(pt, insert=insert, cursor=cur) for pt in points2]
        labels = [(num1, num2, BaseBoxesDB._get_label_dump(l, label_names)) for num1, num2, l in zip(nums1, nums2, labels)]
        cur.executemany('insert into labels (point1, point2, label) values (?,?,?)', labels)
        self.con.commit()

    def get_all_pairs(self, label_names):
        points1 = []
        points2 = []
        labels = []
        with self.con as con:
            script = 'select p1.side as p1_side, p1.sample_num as p1_sample_num, p1.slice_no as p1_slice_no, p1.data as p1_data, p2.side as p2_side, p2.sample_num as p2_sample_num, p2.slice_no as p2_slice_no, p2.data as p2_data, label from labels inner join points as p1 on p1.point_id = labels.point1 inner join points as p2 on p2.point_id = labels.point2 order by labels.rowid asc'
            for row in con.execute(script):
                point1 = {'side': row['p1_side'], 'sample_num': row['p1_sample_num'], 'slice_no': row['p1_slice_no']}
                point1.update(json.loads(row['p1_data']))
                point2 = {'side': row['p2_side'], 'sample_num': row['p2_sample_num'], 'slice_no': row['p2_slice_no']}
                point2.update(json.loads(row['p2_data']))
                label = BaseBoxesDB._get_label_from_dump(row['label'], label_names)
                points1.append(point1)
                points2.append(point2)
                labels.append(label)
        return points1, points2, np.asarray(labels)


class SeedBoxesDB(BaseBoxesDB):
    def __init__(self, fname):
        super(SeedBoxesDB, self).__init__(fname)
        self.log = logging.getLogger('SeedBoxesDB')
        self.log.info('Initializing {}'.format(fname))
        self.log.info('Creating points, seeds and labels table')
        self.create_points_table()
        self.create_seeds_table()
        self.create_labels_table(point1_table='seeds', point2_table='points')
        self.boxes_table = 'main'
        
    # --- getting / inserting data ---
    def insert_pair(self, point1, point2, label, label_names, insert=True):
        num1 = self.get_seed_id(point1, insert=insert)
        num2 = self.get_point_id(point2, insert=insert)
        if not insert:
            assert num1 is not None and num2 is not None, "points for inserting label do not exist!"
        self.insert_label(num1, num2, label, label_names)

    def batch_insert_pair(self, points1, points2, labels, label_names, insert=True):
        # only works if labels are not yet present in the db
        #point_select_query = 'select point_id from points where side = ? and sample_num = ?'
        #seed_select_query = 'select point_id from seeds where side = ? and area = ? and seed_num = ?'
        #label_insert_query = 'insert into labels (point1, point2, label) values (?,?,?)'
        cur = self.con.cursor()
        nums1 = [self.get_seed_id(pt, insert=insert) for pt in points1]
        nums2 = [self.get_point_id(pt, insert=insert) for pt in points2]
        labels = [(num1, num2, BaseBoxesDB._get_label_dump(l, label_names)) for num1, num2, l in zip(nums1, nums2, labels)]
        cur.executemany('insert into labels (point1, point2, label) values (?,?,?)', labels)
        self.con.commit()

    def get_all_pairs(self, label_names, side):
        query = '''select points.point_id as sample_id, labels.label as gt_label
                   from labels
                   inner join points on points.point_id = labels.point2
                   inner join seeds on seeds.point_id = labels.point1
                   where seeds.seed_num = ? and seeds.area = ? and seeds.side = ? and points.side = ? order by points.sample_num asc'''
        seeds, _ = self.get_all_seeds(side, with_label=True)  # sorted seeds
        samples, sample_ids = self.get_all_points(side)  # sorted samples
        gt_labels = np.zeros((len(samples), len(seeds), len(label_names)), dtype=np.float32)
        with self.con as con:
            for i, seed in enumerate(seeds):
                for row in con.execute(query,(seed['seed_num'], seed['area'], str(side), str(side))):
                    gt_label = BaseBoxesDB._get_label_from_dump(row['gt_label'], label_names)
                    idx = sample_ids.index(row['sample_id'])
                    gt_labels[idx,i,:] = gt_label
        return seeds, samples, gt_labels


class PredictionBoxesDB(SeedBoxesDB):
    def __init__(self, fname, boxes_file=None):
        super(SeedBoxesDB, self).__init__(fname)
        self.log = logging.getLogger('PredictionSeedBoxesDB')
        self.log.info('Initializing {}, boxes_file: {}'.format(fname, boxes_file))
        if boxes_file is not None:
            rel_boxes_file = os.path.relpath(boxes_file, os.path.dirname(fname))
            cur_boxes_file = self.get_boxes_file()
            if cur_boxes_file != rel_boxes_file:
                if cur_boxes_file is not None:
                    self.log.warning('Updating existing boxes file {} to {}'.format(cur_boxes_file, rel_boxes_file))
                self.update_boxes_file(rel_boxes_file)
        boxes_file = self.get_boxes_file()
        assert boxes_file is not None, 'Existing prediction database needs boxes_file table'
        boxes_file = os.path.join(os.path.dirname(self.fname), boxes_file)
        self.log.info('Conneting to boxes file db {}'.format(boxes_file))
        with self.con as con:
            con.execute('attach database ? as boxes',(boxes_file, ))
        self.boxes_file = boxes_file
        self.log.info('Creating labels table')
        self.create_labels_table(with_foreign_key=False)
        self.log.info('Creating features table')
        self.create_features_table()
        self.points_table = 'boxes'

    # --- getting / inserting data ---
    def insert_pair(self, point1, point2, label, label_names):
        super(PredictionBoxesDB, self).insert_pair(point1, point2, label, label_names, insert=False)

    def batch_insert_pair(self, points1, points2, labels, label_names):
        super(PredictionBoxesDB, self).batch_insert_pair(points1, points2, labels, label_names, insert=False)

    def insert_feature(self, point, feature, label_names):
        if 'seed_num' in point:
            num = self.get_seed_id(point, insert=False)
            is_seed = 1
        else:
            num = self.get_point_id(point, insert=False)
            is_seed = 0
        assert num is not None, "point is not in DB!"
        query = 'insert into features (point, is_seed, feature) values (?,?,?)'
        with self.con as con:
            feature_str = BaseBoxesDB._get_label_dump([feature], label_names)
            try:
                con.execute(query, (num, is_seed, feature_str))
            except sqlite3.IntegrityError:
                self.log.info('Feature for {} already exists. Not inserting!'.format(num))

    def get_feature(self, point, label_names):
        if 'seed_num' in point:
            num = self.get_seed_id(point, insert=False)
        else:
            num = self.get_point_id(point, insert=False)
        assert num is not None, "point is not in DB!"
        query = 'select feature from features where point = ?'
        with self.con as con:
            feature_str = con.execute(query,(num,)).fetchone()['feature']
        return BaseBoxesDB._get_label_from_dump(feature_str, label_names)[0]

    def get_features(self, points, label_names, is_seed=False):
        if is_seed:
            nums = [self.get_seed_id(pt, insert=False) for pt in points]
        else:
            nums = [self.get_point_id(pt, insert=False) for pt in points]
        with self.con as con:
            features = [0]*len(nums)
            num_rows = 0
            # work around max sql variable limit of sqlite
            for start in range(0, len(nums), 500):
                sl = slice(start, start+500)
                params = [int(is_seed)]+nums[sl]
                query = 'select point, feature from features where is_seed = ? and point in (%s)'%(','.join('?'*(len(params)-1)))
                for row in con.execute(query, params):
                    feat = BaseBoxesDB._get_label_from_dump(row['feature'], label_names)[0]
                    features[nums.index(row['point'])] = feat
                    num_rows += 1
        self.log.info('Got {} features (is_seed {}) for {} points'.format(num_rows, is_seed, len(points)))
        return features

    def get_all_pairs(self, label_names, side):
        query = '''select points.point_id as sample_id, main.labels.label as pred_label, boxes.labels.label as gt_label
                   from main.labels
                   inner join points on points.point_id = main.labels.point2
                   inner join seeds on seeds.point_id = main.labels.point1
                   inner join boxes.labels on points.point_id = boxes.labels.point2 and seeds.point_id = boxes.labels.point1
                   where seeds.seed_num = ? and seeds.area = ? and seeds.side = ? and points.side = ? order by points.sample_num asc'''
        seeds, _ = self.get_all_seeds(side, with_label=True)  # sorted seeds
        samples, sample_ids = self.get_all_points(side)  # sorted samples
        pred_labels = np.zeros((len(samples), len(seeds), len(label_names)), dtype=np.float32)
        gt_labels = np.zeros_like(pred_labels)
        with self.con as con:
            for i, seed in enumerate(seeds):
                for row in con.execute(query,(seed['seed_num'], seed['area'], str(side), str(side))):
                    pred_label = BaseBoxesDB._get_label_from_dump(row['pred_label'], label_names)
                    gt_label = BaseBoxesDB._get_label_from_dump(row['gt_label'], label_names)
                    idx = sample_ids.index(row['sample_id'])
                    pred_labels[idx,i,:] = pred_label
                    gt_labels[idx,i,:] = gt_label
        return seeds, samples, gt_labels, pred_labels

    def get_boxes_file(self):
        try:
            with self.con as con:
                boxes_file = con.execute('select fname from boxes_file order by rowid desc').fetchone()['fname']
                return boxes_file
        except (sqlite3.OperationalError, TypeError):
            return None

    def update_boxes_file(self, boxes_file):
        with self.con as con:
            con.execute('create table if not exists boxes_file (fname text)')
            if self.get_boxes_file() is not None:
                con.execute('update boxes_file set fname = ?', (boxes_file,))
            else:
                con.execute('insert into boxes_file(fname) values (?)', (boxes_file,))

    def create_features_table(self):
        script = 'create table if not exists features (point integer, is_seed integer, feature text)'
        with self.con as con:
            con.execute(script)

 
