#!/usr/bin/env python
# syndb.py - paradigm/syncretism database from ods spreadsheet tables

"""Morphological paradigm/syncretism database from ODS spreadsheet tables."""

from __future__ import unicode_literals, print_function

import contextlib
import csv
import io
import itertools
import os
import sys
import zipfile
import xml.etree.cElementTree as etree

PY2 = (sys.version_info.major == 2)

if PY2:
    from itertools import izip as zip

import sqlalchemy as sa
import sqlalchemy.orm
import sqlalchemy.ext.declarative

from sqlalchemy import Column, Integer, Unicode
from sqlalchemy.orm import relationship

__title__ = 'syndb.py'
__author__ = 'Sebastian Bank <sebastian.bank@uni-leipzig.de>'
__license__ = 'MIT, see LICENSE.txt'
__copyright__ = 'Copyright (c) 2013,2017 Sebastian Bank'

ODS_FILE = 'Datenbank2.ods'

NAMESPACES = {'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
              'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'}

DB_FILE = 'syndb.sqlite3'

ENGINE = sa.create_engine('sqlite:///%s' % DB_FILE, echo=False)


def get_content_tree(filename=ODS_FILE, content='content.xml'):
    with zipfile.ZipFile(filename) as archive:
        result = etree.parse(archive.open(content))
    return result


def load_tables(tree, ns=NAMESPACES):
    result = {}
    for table in tree.iterfind('office:body/office:spreadsheet/table:table', ns):
        name = table.attrib['{%(table)s}name' % ns]
        rows = []
        for r in table.iterfind('table:table-row', ns):
            cols = []
            for c in r.iterfind('table:table-cell', ns):
                n = int(c.attrib.get('{%(table)s}number-columns-repeated' % ns, '1'))
                text = etree.tostring(c, 'utf-8', method='text').decode('utf-8')
                cols.extend([text] * n)

            if any(cols):
                rows.append(tuple(cols))
        result[name] = rows
    return result


@sa.event.listens_for(sa.engine.Engine, 'connect')
def set_sqlite_pragma(dbapi_conn, connection_record):
    with contextlib.closing(dbapi_conn.cursor()) as cursor:
        cursor.execute('PRAGMA foreign_keys = ON')


class BooleanZeroOne(sa.TypeDecorator):

    impl = sa.Boolean

    def process_bind_param(self, value, dialect):
        if value in ('0', '1'):
            value = int(value)
        return value


Base = sa.ext.declarative.declarative_base()


def dbschema(metadata=Base.metadata, engine=ENGINE):
    def dump(sql):
        print(sql.compile(dialect=engine.dialect))

    dumper = sa.create_engine(engine.url, strategy='mock', executor=dump)
    metadata.create_all(dumper, checkfirst=False)


def dump_sql(engine=ENGINE, encoding='utf-8'):
    filename = '%s.sql' % os.path.splitext(engine.url.database)[0]
    with contextlib.closing(engine.raw_connection()) as dbapi_conn,\
         io.open(filename, 'w', encoding=encoding) as f:
        for line in dbapi_conn.iterdump():
            f.write('%s\n' % line)
    return filename


Session = sa.orm.sessionmaker(bind=ENGINE)


class Language(Base):

    __tablename__ = 'language'

    iso = Column(Unicode(3), sa.CheckConstraint('length(iso) = 3'), primary_key=True)

    name = Column(Unicode, sa.CheckConstraint("name != ''"), nullable=False)

    family = Column(Unicode)
    stock = Column(Unicode)
    country = Column(Unicode)
    area = Column(Unicode)
    speakers = Column(Integer, sa.CheckConstraint('speakers >= 0'))

    paradigms = relationship('Paradigm', back_populates='language',
                             order_by='Paradigm.name')


class ParadigmClass(Base):

    __tablename__ = 'cls'

    id = Column(Integer, primary_key=True)

    name = Column(Unicode, sa.CheckConstraint("name != ''"), nullable=False, unique=True)

    ncells = Column(Integer, nullable=False)
    nrows = Column(Integer, sa.CheckConstraint('nrows > 0'), nullable=False)
    ncols = Column(Integer, sa.CheckConstraint('ncols > 0'), nullable=False)

    __table_args__ = (
        sa.CheckConstraint('ncells = nrows * ncols'),
    )

    cells = relationship('ParadigmClassCell', back_populates='cls',
                         order_by='(ParadigmClassCell.row,'
                                  ' ParadigmClassCell.col)')

    paradigms = relationship('Paradigm', back_populates='cls',
                             order_by='(Paradigm.iso, Paradigm.name)')


class ParadigmClassCell(Base):

    __tablename__ = 'clscell'

    cls_id = Column(sa.ForeignKey('cls.id'), primary_key=True)
    index = Column(Integer, sa.CheckConstraint('"index" > 0'), primary_key=True)

    row = Column(Integer, sa.CheckConstraint('"row" > 0'), nullable=False)
    col = Column(Integer, sa.CheckConstraint('col > 0'), nullable=False)

    blind = Column(BooleanZeroOne, nullable=False, default=False)

    label = Column(Unicode, sa.CheckConstraint("label != ''"), nullable=False)

    case = Column(Unicode)
    number = Column(Unicode)
    definiteness = Column(Unicode)
    person = Column(Unicode)
    case_spec = Column(Unicode)
    number_spec = Column(Unicode)
    person_spec = Column(Unicode)

    __table_args__ = (
        sa.UniqueConstraint(cls_id, row, col),
        sa.UniqueConstraint(cls_id, label),
    )

    cls = relationship('ParadigmClass')


class Reference(Base):

    __tablename__ = 'reference'

    bibkey = Column(Unicode(3), sa.CheckConstraint("bibkey != ''"), primary_key=True)

    entry = Column(Unicode, sa.CheckConstraint("entry != ''"), nullable=False)


class Paradigm(Base):

    __tablename__ = 'paradigm'

    id = Column(Integer, primary_key=True)

    iso = Column(sa.ForeignKey('language.iso'), nullable=False)
    cls_id = Column(sa.ForeignKey('cls.id'), nullable=False)
    name = Column(Unicode, sa.CheckConstraint("name != ''"), nullable=False)

    stem = Column(Unicode, sa.CheckConstraint("stem != ''"), nullable=False)
    gloss = Column(Unicode, sa.CheckConstraint("gloss != ''"), nullable=False)
    reference_bibkey = Column(sa.ForeignKey('reference.bibkey'))
    pages = Column(Unicode)

    __table_args__ = (
        sa.UniqueConstraint(iso, name),
    )

    language = relationship('Language', back_populates='paradigms')

    cls = relationship('ParadigmClass', back_populates='paradigms')

    contents = relationship('ParadigmContent', back_populates='paradigm',
                            order_by='(ParadigmContent.cell_index,'
                                     ' ParadigmContent.position)')

    syncretisms = relationship('Syncretism', back_populates='paradigm',
                               order_by='Syncretism.form')


class ParadigmContent(Base):

    __tablename__ = 'paradigmcontent'

    paradigm_id = Column(sa.ForeignKey('paradigm.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)
    position = Column(Integer, primary_key=True)

    form = Column(Unicode, sa.CheckConstraint("form != ''"), nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic'), nullable=False)

    __table_args__ = (
        sa.ForeignKeyConstraint([cell_cls, cell_index],
                                ['clscell.cls_id', 'clscell.index']),
        sa.CheckConstraint("(position = 0) OR (kind != 'stem')"),
        #sa.CheckConstraint("(position = 0) = (kind = 'stem')"),
    )

    paradigm = relationship('Paradigm', back_populates='contents')

    cell = relationship('ParadigmClassCell')


class Syncretism(Base):

    __tablename__ = 'syncretism'

    id = Column(Integer, primary_key=True)

    paradigm_id = Column(sa.ForeignKey('paradigm.id'))

    form = Column(Unicode, sa.CheckConstraint("form != ''"), nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic'), nullable=False)

    paradigm = relationship('Paradigm', back_populates='syncretisms')

    cells = relationship('SyncretismCell', back_populates='syncretism',
                         order_by='SyncretismCell.cell_index')


class SyncretismCell(Base):

    __tablename__ = 'syncretismcell'

    syncretism_id = Column(sa.ForeignKey('syncretism.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)

    __table_args__ = (
        sa.ForeignKeyConstraint([cell_cls, cell_index],
                                ['clscell.cls_id', 'clscell.index']),
    )

    syncretism = relationship('Syncretism', back_populates='cells')

    cell = relationship('ParadigmClassCell')


def insert_tables(tables, engine=ENGINE):
    if os.path.exists(engine.url.database):
        os.remove(engine.url.database)

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for cls in [Reference, Language, ParadigmClass, ParadigmClassCell,
                    Paradigm, ParadigmContent, Syncretism, SyncretismCell]:
            table = tables[cls.__tablename__]
            header = [h for h in table[0] if h]
            rows = ((v if v.strip() else None for v in row) for row in table[1:])
            params = [dict(zip(header, r)) for r in rows]
            conn.execute(sa.insert(cls), params)


def export_csv(metadata=Base.metadata, engine=ENGINE, encoding='utf-8'):
    filename = '%s.zip' % os.path.splitext(engine.url.database)[0]
    with engine.connect() as conn,\
         zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as archive:
        for table in metadata.sorted_tables:
            result = conn.execute(table.select())
            with (io.BytesIO() if PY2 else io.StringIO()) as f:
                writer = csv.writer(f)
                writer.writerow(result.keys())
                if PY2:
                    for row in result:
                        srow = [v.encode(encoding) if isinstance(v, unicode) else v
                                for v in row]
                        writer.writerow(srow)
                    data = f.getvalue()
                else:
                    writer.writerows(result)
                    data = f.getvalue().encode(encoding)
            archive.writestr('%s.csv' % table.name, data)
    return filename


def render_html(filename='paradigms.html', encoding='utf-8'):
    with contextlib.closing(Session()) as session,\
         io.open(filename, 'w', encoding=encoding) as f:
        query = session.query(Paradigm).join('cls')\
            .options(sa.orm.contains_eager('cls'),
                     sa.orm.subqueryload('cls', 'cells'),
                     sa.orm.subqueryload('contents'))
        f.write('\n'.join(['<!doctype html>', '<html>',
                           '<head><meta charset="%s"></head>' % encoding,
                           '<body>', '']))
        for p in query:
            f.write(html_paradigm(p))
            f.write('\n\n')
        f.write('\n'.join(['</body>', '</html>']))


def html_paradigm(paradigm):
    result = [
        '<h2>%s %s (%s)</h2>' % (paradigm.iso, paradigm.name, paradigm.cls.name),
        '<table border="1">'
    ]
    contents = {cell: list(occ) for cell, occ in
                itertools.groupby(paradigm.contents, key=lambda c: c.cell)}
    cmp = lambda a, b: (a > b) - (a < b)
    for row, cells in itertools.groupby(paradigm.cls.cells, key=lambda c: c.row):
        result.append('<tr>')
        for cell in cells:
            result.append('<th>%s</th>' % cell.label)
            if cell.blind:
                result.append('<td style="background-color:#ddd"></td>')
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
                result.append('<td>%s</td>' % (forms))
        result.append('</tr>')
    result.append('</table>')
    return '\n'.join(result)


if __name__ == '__main__':
    #dbschema()
    tables = load_tables(get_content_tree())
    insert_tables(tables)
    dump_sql()
    export_csv()
    render_html()
