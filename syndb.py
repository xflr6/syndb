#!/usr/bin/env python3

"""Morphological paradigm/syncretism database from ODS spreadsheet tables."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import contextlib
import csv
import datetime
import io
import itertools
import os
import pathlib
import sys
from typing import Union
import xml.etree.ElementTree as etree
import zipfile

import sqlalchemy as sa
from sqlalchemy import Column, Integer, String
import sqlalchemy.orm
from sqlalchemy.orm import relationship

__title__ = 'syndb.py'
__author__ = 'Sebastian Bank <sebastian.bank@uni-leipzig.de>'
__license__ = 'MIT, see LICENSE.txt'
__copyright__ = 'Copyright (c) 2013,2017 Sebastian Bank'

ODS_FILE = pathlib.Path('Datenbank2.ods')

NAMESPACES = {'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
              'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'}

DB_PATH = pathlib.Path('syndb.sqlite3')

ENGINE = sa.create_engine(f'sqlite:///{DB_PATH}', echo=False)

REGISTRY = sa.orm.registry()

PARADIGMS_HTML = pathlib.Path('paradigms.html')

ENCODING = 'utf-8'


def get_content_tree(filename: Union[os.PathLike, str] = ODS_FILE, *,
                     content: str = 'content.xml') -> etree.ElementTree:
    with zipfile.ZipFile(filename) as archive, archive.open(content) as f:
        result = etree.parse(f)
    return result


def get_element_text(element: etree.Element) -> str:
    return etree.tostring(element, 'utf-8', method='text').decode('utf-8')


def load_tables(tree: etree.ElementTree, *,
                ns: Mapping[str, str] = NAMESPACES
                ) -> dict[str, list[tuple[str, ...]]]:
    ns_table = '{{{table}}}'.format_map(ns)

    def iterrows(table: etree.Element) -> Iterator[tuple[str, ...]]:
        for r in table.iterfind('table:table-row', ns):
            cols = []
            for c in r.iterfind('table:table-cell', ns):
                n = int(c.attrib.get(f'{ns_table}number-columns-repeated', '1'))
                cols.extend([get_element_text(c)] * n)

            if any(cols):
                yield tuple(cols)

    tables = tree.iterfind('office:body/office:spreadsheet/table:table', ns)
    return {t.attrib[f'{ns_table}name']: list(iterrows(t)) for t in tables}


@sa.event.listens_for(sa.engine.Engine, 'connect')
def set_sqlite_pragma(dbapi_conn, connection_record) -> None:
    with contextlib.closing(dbapi_conn.cursor()) as cursor:
        cursor.execute('PRAGMA foreign_keys = ON')


class BooleanZeroOne(sa.TypeDecorator):

    impl = sa.Boolean

    def process_bind_param(self, value, dialect):
        if value in ('0', '1'):
            value = int(value)
        return value


def dbschema(metadata: sa.MetaData = REGISTRY.metadata, *,
             engine: sa.engine.Engine = ENGINE) -> None:
    def dump(sql):
        print(sql.compile(dialect=engine.dialect))

    mock_engine = sa.create_mock_engine(engine.url, executor=dump)
    metadata.create_all(mock_engine, checkfirst=False)


def dump_sql(engine: sa.engine.Engine = ENGINE, *,
             encoding: str = ENCODING) -> pathlib.Path:
    filepath = pathlib.Path(engine.url.database).with_suffix('.sql')
    with contextlib.closing(engine.raw_connection()) as dbapi_conn,\
         filepath.open('w', encoding=encoding) as f:
        for line in dbapi_conn.iterdump():
            print(line, file=f)
    return filepath


Session = sa.orm.sessionmaker(bind=ENGINE)


@REGISTRY.mapped
class Language:

    __tablename__ = 'language'

    iso = Column(String(3), sa.CheckConstraint('length(iso) = 3'),
                 primary_key=True)

    name = Column(String, sa.CheckConstraint("name != ''"), nullable=False)

    family = Column(String)
    stock = Column(String)
    country = Column(String)
    area = Column(String)
    speakers = Column(Integer, sa.CheckConstraint('speakers >= 0'))

    paradigms = relationship('Paradigm', back_populates='language',
                             order_by='Paradigm.name')


@REGISTRY.mapped
class ParadigmClass:

    __tablename__ = 'cls'

    id = Column(Integer, primary_key=True)

    name = Column(String, sa.CheckConstraint("name != ''"), nullable=False, unique=True)

    ncells = Column(Integer, nullable=False)
    nrows = Column(Integer, sa.CheckConstraint('nrows > 0'), nullable=False)
    ncols = Column(Integer, sa.CheckConstraint('ncols > 0'), nullable=False)

    __table_args__ = (sa.CheckConstraint('ncells = nrows * ncols'),)

    cells = relationship('ParadigmClassCell', back_populates='cls',
                         order_by='(ParadigmClassCell.row,'
                                  ' ParadigmClassCell.col)')

    paradigms = relationship('Paradigm', back_populates='cls',
                             order_by='(Paradigm.iso, Paradigm.name)')


@REGISTRY.mapped
class ParadigmClassCell:

    __tablename__ = 'clscell'

    cls_id = Column(sa.ForeignKey('cls.id'), primary_key=True)
    index = Column(Integer, sa.CheckConstraint('"index" > 0'), primary_key=True)

    row = Column(Integer, sa.CheckConstraint('"row" > 0'), nullable=False)
    col = Column(Integer, sa.CheckConstraint('col > 0'), nullable=False)

    blind = Column(BooleanZeroOne(create_constraint=True), nullable=False,
                   default=False)

    label = Column(String, sa.CheckConstraint("label != ''"), nullable=False)

    case = Column(String)
    number = Column(String)
    definiteness = Column(String)
    person = Column(String)
    case_spec = Column(String)
    number_spec = Column(String)
    person_spec = Column(String)

    __table_args__ = (sa.UniqueConstraint(cls_id, row, col),
                      sa.UniqueConstraint(cls_id, label))

    cls = relationship('ParadigmClass', innerjoin=True)


@REGISTRY.mapped
class Reference:

    __tablename__ = 'reference'

    bibkey = Column(String(3), sa.CheckConstraint("bibkey != ''"),
                    primary_key=True)

    entry = Column(String, sa.CheckConstraint("entry != ''"),
                   nullable=False)


@REGISTRY.mapped
class Paradigm:

    __tablename__ = 'paradigm'

    id = Column(Integer, primary_key=True)

    iso = Column(sa.ForeignKey('language.iso'), nullable=False)
    cls_id = Column(sa.ForeignKey('cls.id'), nullable=False)
    name = Column(String, sa.CheckConstraint("name != ''"), nullable=False)

    stem = Column(String, sa.CheckConstraint("stem != ''"), nullable=False)
    gloss = Column(String, sa.CheckConstraint("gloss != ''"), nullable=False)
    reference_bibkey = Column(sa.ForeignKey('reference.bibkey'))
    pages = Column(String)

    __table_args__ = (sa.UniqueConstraint(iso, name),)

    language = relationship('Language', innerjoin=True, back_populates='paradigms')

    cls = relationship('ParadigmClass', innerjoin=True, back_populates='paradigms')

    contents = relationship('ParadigmContent', back_populates='paradigm',
                            order_by='(ParadigmContent.cell_index,'
                                     ' ParadigmContent.position)')

    syncretisms = relationship('Syncretism', back_populates='paradigm',
                               order_by='Syncretism.form')


@REGISTRY.mapped
class ParadigmContent:

    __tablename__ = 'paradigmcontent'

    paradigm_id = Column(sa.ForeignKey('paradigm.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)
    position = Column(Integer, primary_key=True)

    form = Column(String, sa.CheckConstraint("form != ''"), nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic', create_constraint=True),
                  nullable=False)

    __table_args__ = (sa.ForeignKeyConstraint([cell_cls, cell_index],
                                              ['clscell.cls_id',
                                               'clscell.index']),
                      sa.CheckConstraint("(position = 0) OR (kind != 'stem')"),)
                      #sa.CheckConstraint("(position = 0) = (kind = 'stem')"),)

    paradigm = relationship('Paradigm', innerjoin=True, back_populates='contents')

    cell = relationship('ParadigmClassCell')


@REGISTRY.mapped
class Syncretism:

    __tablename__ = 'syncretism'

    id = Column(Integer, primary_key=True)

    paradigm_id = Column(sa.ForeignKey('paradigm.id'))

    form = Column(String, sa.CheckConstraint("form != ''"), nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic', create_constraint=True),
                  nullable=False)

    paradigm = relationship('Paradigm', innerjoin=True, back_populates='syncretisms')

    cells = relationship('SyncretismCell', back_populates='syncretism',
                         order_by='SyncretismCell.cell_index')


@REGISTRY.mapped
class SyncretismCell:

    __tablename__ = 'syncretismcell'

    syncretism_id = Column(sa.ForeignKey('syncretism.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)

    __table_args__ = (sa.ForeignKeyConstraint([cell_cls, cell_index],
                                              ['clscell.cls_id',
                                               'clscell.index']),)

    syncretism = relationship('Syncretism', innerjoin=True, back_populates='cells')

    cell = relationship('ParadigmClassCell')


def insert_tables(tables: dict[str, Sequence[Sequence[str]]], *,
                  engine: sa.engine.Engine = ENGINE) -> None:
    db_path = pathlib.Path(engine.url.database)
    if db_path.exists():
        db_path.unlink()

    REGISTRY.metadata.create_all(engine)
    with engine.begin() as conn:
        for cls in [Reference, Language, ParadigmClass, ParadigmClassCell,
                    Paradigm, ParadigmContent, Syncretism, SyncretismCell]:
            header, *rows = tables[cls.__tablename__]
            header = [h for h in header if h]
            rows = ((v.strip() or None for v in row) for row in rows)
            params = [dict(zip(header, r)) for r in rows]
            conn.execute(sa.insert(cls), params)


def export_csv(metadata: sa.MetaData = REGISTRY.metadata, *,
               engine: sa.engine.Engine = ENGINE,
               encoding: str = ENCODING) -> pathlib.Path:
    filepath = pathlib.Path(engine.url.database).with_suffix('.zip')
    # zipfile only supports naive, use local time
    date_time = datetime.datetime.now().timetuple()[:6]
    with engine.connect() as conn, zipfile.ZipFile(filepath, 'w') as archive:
        for table in sorted(metadata.sorted_tables, key=lambda x: x.name):
            result = conn.execute(table.select())
            info = zipfile.ZipInfo(f'{table.name}.csv', date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            with io.TextIOWrapper(archive.open(info, 'w'),
                                  encoding=encoding, newline='',
                                  line_buffering=True) as f:
                writer = csv.writer(f)
                writer.writerow(result.keys())
                writer.writerows(result)
    return filepath


def render_html(filepath: pathlib.Path = PARADIGMS_HTML, *,
                encoding: str = ENCODING) -> None:
    query = (sa.select(Paradigm)
             .options(sa.orm.joinedload(Paradigm.cls).selectinload(ParadigmClass.cells),
                      sa.orm.selectinload(Paradigm.contents)))
    with Session() as session, filepath.open('w', encoding=encoding) as f:
        paradigms = session.execute(query).scalars()
        for line in iterhtml(paradigms, encoding=encoding):
            print(line, file=f)


def iterhtml(paradigms: Iterable[Paradigm], *, encoding: str) -> Iterator[str]:
    yield '<!doctype html>'
    yield '<html>'
    yield f'<head><meta charset="{encoding}"></head>'
    yield '<body>'
    for p in paradigms:
        yield from iterlines(p)
        yield ''
    yield '</body>'
    yield '</html>'


def iterlines(paradigm: Paradigm) -> Iterator[str]:
    yield f'<h2>{paradigm.iso} {paradigm.name} ({paradigm.cls.name})</h2>'
    yield '<table border="1">'
    contents = {cell: list(occ) for cell, occ in
                itertools.groupby(paradigm.contents, key=lambda c: c.cell)}
    cmp = lambda a, b: (a > b) - (a < b)
    for row, cells in itertools.groupby(paradigm.cls.cells, key=lambda c: c.row):
        yield'<tr>'
        for cell in cells:
            yield f'<th>{cell.label}</th>'
            if cell.blind:
                yield '<td style="background-color:#ddd"></td>'
            else:
                occ = contents.get(cell, [])
                slots = {kind: list(occ) for kind, occ in
                         itertools.groupby(occ, key=lambda o: cmp(o.position, 0))}
                pf = '-'.join(o.form for o in slots.get(-1, []))
                if 0 in slots:
                    assert len(slots[0]) == 1
                    st = slots[0][0].form
                else:
                    st = paradigm.stem
                sf = '-'.join(o.form for o in slots.get(1, []))
                forms = '-'.join(s for s in [pf, st, sf] if s)
                yield f'<td>{forms}s</td>'
        yield '</tr>'
    yield '</table>'


def main() -> None:
    #dbschema()
    tree = get_content_tree()
    tables = load_tables(tree)
    insert_tables(tables)
    dump_sql()
    export_csv()
    render_html()
    return None


if __name__ == '__main__':
    sys.exit(main())
