Syndb
=====

This Python 2/3 script loads a hand-curated database of morphological paradigms
and syncretisms from an `ODS spreadsheet`_ with multiple sheets into SQLite_
and exports (converts) the database into the following files:

- a ``syndb.sql`` plain-text SQL dump
- a ``syndb.zip`` archive containing all tables as CSV_
- a ``paradigms.html`` file showing all paradigms in their tabular layout


Dependencies
------------

- Python_ 2.7 or 3.4+ (including ElementTree_ used for ODS reading)
- SQLAlchemy_


.. _ODS spreadsheet: https://en.wikipedia.org/wiki/OpenDocument
.. _SQLite: https://www.sqlite.org
.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values
.. _Python: https://www.python.org
.. _ElementTree: https://docs.python.org/library/xml.etree.elementtree.html
.. _SQLAlchemy: https://www.sqlalchemy.org/
