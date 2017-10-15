#!/usr/bin/env python
# syndb.py - paradigm/syncretism database from ods spreadsheet tables

"""Morphological paradigm/syncretism database from ODS spreadsheet tables."""

from __future__ import unicode_literals, print_function

import io
import os
import csv
import sys
import zipfile
import itertools
import contextlib
import xml.etree.cElementTree as etree

import sqlalchemy as sa
import sqlalchemy.orm
import sqlalchemy.ext.declarative
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, backref

__author__ = 'Sebastian Bank <sebastian.bank@uni-leipzig.de>'
__license__ = 'MIT, see LICENSE'
__copyright__ = 'Copyright (c) 2013,2017 Sebastian Bank'

ODS_FILE = 'Datenbank2.ods'
DB_FILE = 'syndb.sqlite3'

PY2 = sys.version_info < (3,)

NAMESPACES = {
    'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
    'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
}


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


class BooleanZeroOne(sa.TypeDecorator):

    impl = sa.Boolean

    def process_bind_param(self, value, dialect):
        if value in ('0', '1'):
            value = int(value)
        return value


Base = sa.ext.declarative.declarative_base()


class Language(Base):

    __tablename__ = 'language'

    iso = Column(String(3), primary_key=True)
    name = Column(String, nullable=False)
    family = Column(String)
    stock = Column(String)
    country = Column(String)
    area = Column(String)
    speakers = Column(Integer)


class ParadigmClass(Base):

    __tablename__ = 'cls'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    ncells = Column(Integer, nullable=False)
    nrows = Column(Integer, nullable=False)
    ncols = Column(Integer, nullable=False)

    __table_args__ = (
        sa.CheckConstraint('ncells = nrows * ncols'),
    )


class ParadigmClassCell(Base):

    __tablename__ = 'clscell'

    cls_id = Column(Integer, ForeignKey('cls.id'), primary_key=True)
    index = Column(Integer, primary_key=True)
    row = Column(Integer, nullable=False)
    col = Column(Integer, nullable=False)
    blind = Column(BooleanZeroOne, nullable=False, default=False)
    label = Column(String, nullable=False)
    case = Column(String)
    number = Column(String)
    definiteness = Column(String)
    person = Column(String)
    case_spec = Column(String)
    number_spec = Column(String)
    person_spec = Column(String)

    __table_args__ = (
        sa.UniqueConstraint(cls_id, row, col),
        sa.UniqueConstraint(cls_id, label),
    )

    cls = relationship(ParadigmClass, backref=backref('cells', order_by=(row, col)))


class Reference(Base):

    __tablename__ = 'reference'

    bibkey = Column(String(3), primary_key=True)
    entry = Column(String)


class Paradigm(Base):

    __tablename__ = 'paradigm'

    id = Column(Integer, primary_key=True)
    iso = Column(String(3), ForeignKey('language.iso'), nullable=False)
    cls_id = Column(Integer, ForeignKey('cls.id'), nullable=False)
    name = Column(String, nullable=False)
    stem = Column(String, nullable=False)
    gloss = Column(String, nullable=False)
    reference_bibkey = Column(String, ForeignKey('reference.bibkey'))
    pages = Column(String)

    __table_args__ = (
        sa.UniqueConstraint(iso, name),
    )

    language = relationship(Language, backref=backref('paradigms', order_by=name))
    cls = relationship(ParadigmClass, backref=backref('paradigms', order_by=(iso, name)))


class ParadigmContent(Base):

    __tablename__ = 'paradigmcontent'

    paradigm_id = Column(Integer, ForeignKey('paradigm.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)
    position = Column(Integer, primary_key=True)
    form = Column(String, nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic'), nullable=False)

    __table_args__ = (
        sa.ForeignKeyConstraint([cell_cls, cell_index], ['clscell.cls_id', 'clscell.index']),
        #sa.CheckConstraint("(position = 0) = (kind = 'stem')"),
    )

    paradigm = relationship(Paradigm, backref=backref('contents', order_by=(cell_index, position)))
    cell = relationship(ParadigmClassCell)


class Syncretism(Base):

    __tablename__ = 'syncretism'

    id = Column(Integer, primary_key=True)
    paradigm_id = Column(Integer, ForeignKey('paradigm.id'))
    form = Column(String, nullable=False)
    kind = Column(sa.Enum('stem', 'affix', 'clitic'), nullable=False)

    paradigm = relationship(Paradigm, backref=backref('syncretisms', order_by=form))


class SyncretismCell(Base):

    __tablename__ = 'syncretismcell'

    syncretism_id = Column(Integer, ForeignKey('syncretism.id'), primary_key=True)
    cell_cls = Column(Integer, primary_key=True)
    cell_index = Column(Integer, primary_key=True)

    __table_args__ = (
        sa.ForeignKeyConstraint([cell_cls, cell_index], ['clscell.cls_id', 'clscell.index']),
    )

    syncretism = relationship(Syncretism, backref=backref('cells', order_by=cell_index))
    cell = relationship(ParadigmClassCell)


engine = sa.create_engine('sqlite:///%s' % DB_FILE, echo=False)


@sa.event.listens_for(sa.engine.Engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
    with contextlib.closing(dbapi_connection.cursor()) as cursor:
        cursor.execute('PRAGMA foreign_keys = ON')


def dbschema(metadata=Base.metadata, engine=engine):
    def dump(sql):
        print(sql.compile(dialect=engine.dialect))
    dumper = sa.create_engine(engine.url, strategy='mock', executor=dump)
    metadata.create_all(dumper, checkfirst=False)


def insert_tables(tables, engine=engine):
    if os.path.exists(engine.url.database):
        os.remove(engine.url.database)
    Base.metadata.create_all(engine)
    models = [Reference, Language, ParadigmClass, ParadigmClassCell, Paradigm, ParadigmContent, Syncretism, SyncretismCell]
    with engine.begin() as conn:
        for cls in models:
            table = tables[cls.__tablename__]
            header = [h for h in table[0] if h]
            rows = ((c if c.strip() else None for c in cols) for cols in table[1:])
            params = [dict(zip(header, cols)) for cols in rows]
            conn.execute(sa.insert(cls), params)


def dump_sql(engine=engine, encoding='utf-8'):
    filename = '%s.sql' % os.path.splitext(engine.url.database)[0]
    with engine.connect() as conn,\
         io.open(filename, 'w', encoding=encoding) as fd:
        for line in conn.connection.iterdump():
            fd.write(('%s\n' % line))


def export_csv(metadata=Base.metadata, engine=engine, encoding='utf-8'):
    filename = '%s.zip' % os.path.splitext(engine.url.database)[0]
    with engine.connect() as conn,\
         zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as archive:
        get_fd = io.BytesIO if PY2 else io.StringIO
        for table in metadata.sorted_tables:
            with contextlib.closing(get_fd()) as fd:
                writer = csv.writer(fd)
                result = conn.execute(table.select())
                writer.writerow(result.keys())
                if PY2:
                    for row in result:
                        writer.writerow([unicode(col).encode(encoding) if col else col for col in row])
                    data = fd.getvalue()
                else:
                    for row in result:
                        writer.writerow(row)
                    data = fd.getvalue().encode(encoding)
                archive.writestr('%s.csv' % table.name, data)


Session = sa.orm.sessionmaker(bind=engine)


def render_html(filename='paradigms.html', encoding='utf-8'):
    with contextlib.closing(Session()) as session,\
         io.open(filename, 'w', encoding=encoding) as fd:
        query = session.query(Paradigm).join('cls').options(
            sa.orm.contains_eager('cls'),
            sa.orm.subqueryload('cls', 'cells'),
            sa.orm.subqueryload('contents'))
        fd.write('\n'.join(['<!doctype html>', '<html>', '<head><meta charset="utf-8"></head>', '<body>', '']))
        for p in query:
            fd.write(html_paradigm(p))
            fd.write('\n\n')
        fd.write('\n'.join(['</body>', '</html>']))


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
                occ = contents.get(cell,[])
                slots= {kind: list(occ) for kind, occ in 
                    itertools.groupby(occ, key=lambda o: cmp(o.position, 0))}
                pf = '-'.join(o.form  for o in slots.get(-1, []))
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
    tables = load_tables(get_content_tree())
    #dbschema()
    insert_tables(tables)
    dump_sql()
    export_csv()
    render_html()
