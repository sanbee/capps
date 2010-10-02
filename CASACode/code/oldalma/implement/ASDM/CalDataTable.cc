
/*
 * ALMA - Atacama Large Millimeter Array
 * (c) European Southern Observatory, 2002
 * (c) Associated Universities Inc., 2002
 * Copyright by ESO (in the framework of the ALMA collaboration),
 * Copyright by AUI (in the framework of the ALMA collaboration),
 * All rights reserved.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY, without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307  USA
 *
 * Warning!
 *  -------------------------------------------------------------------- 
 * | This is generated code!  Do not modify this file.                  |
 * | If you do, all changes will be lost when the file is re-generated. |
 *  --------------------------------------------------------------------
 *
 * File CalDataTable.cpp
 */
#include <ConversionException.h>
#include <DuplicateKey.h>
#include <OutOfBoundsException.h>

using asdm::ConversionException;
using asdm::DuplicateKey;
using asdm::OutOfBoundsException;

#include <ASDM.h>
#include <CalDataTable.h>
#include <CalDataRow.h>
#include <Parser.h>

using asdm::ASDM;
using asdm::CalDataTable;
using asdm::CalDataRow;
using asdm::Parser;

#include <iostream>
#include <sstream>
#include <set>
using namespace std;

#include <Misc.h>
using namespace asdm;


namespace asdm {

	string CalDataTable::tableName = "CalData";
	

	/**
	 * The list of field names that make up key key.
	 * (Initialization is in the constructor.)
	 */
	vector<string> CalDataTable::key;

	/**
	 * Return the list of field names that make up key key
	 * as an array of strings.
	 */	
	vector<string> CalDataTable::getKeyName() {
		return key;
	}


	CalDataTable::CalDataTable(ASDM &c) : container(c) {

	
		key.push_back("calDataId");
	


		// Define a default entity.
		entity.setEntityId(EntityId("uid://X0/X0/X0"));
		entity.setEntityIdEncrypted("na");
		entity.setEntityTypeName("CalDataTable");
		entity.setEntityVersion("1");
		entity.setInstanceVersion("1");
		
		// Archive XML
		archiveAsBin = false;
		
		// File XML
		fileAsBin = false;
	}
	
/**
 * A destructor for CalDataTable.
 */
 
	CalDataTable::~CalDataTable() {
		for (unsigned int i = 0; i < privateRows.size(); i++) 
			delete(privateRows.at(i));
	}

	/**
	 * Container to which this table belongs.
	 */
	ASDM &CalDataTable::getContainer() const {
		return container;
	}

	/**
	 * Return the number of rows in the table.
	 */

	unsigned int CalDataTable::size() {
		return row.size();
	}	
	
	
	/**
	 * Return the name of this table.
	 */
	string CalDataTable::getName() const {
		return tableName;
	}

	/**
	 * Return this table's Entity.
	 */
	Entity CalDataTable::getEntity() const {
		return entity;
	}

	/**
	 * Set this table's Entity.
	 */
	void CalDataTable::setEntity(Entity e) {
		this->entity = e; 
	}
	
	//
	// ====> Row creation.
	//
	
	/**
	 * Create a new row.
	 */
	CalDataRow *CalDataTable::newRow() {
		return new CalDataRow (*this);
	}
	
	CalDataRow *CalDataTable::newRowEmpty() {
		return newRow ();
	}


	/**
	 * Create a new row initialized to the specified values.
	 * @return a pointer on the created and initialized row.
	
 	 * @param numScan. 
	
 	 * @param scanSet. 
	
 	 * @param calType. 
	
 	 * @param startTimeObserved. 
	
 	 * @param endTimeObserved. 
	
 	 * @param calDataType. 
	
     */
	CalDataRow* CalDataTable::newRow(int numScan, vector<int > scanSet, CalTypeMod::CalType calType, ArrayTime startTimeObserved, ArrayTime endTimeObserved, CalDataOriginMod::CalDataOrigin calDataType){
		CalDataRow *row = new CalDataRow(*this);
			
		row->setNumScan(numScan);
			
		row->setScanSet(scanSet);
			
		row->setCalType(calType);
			
		row->setStartTimeObserved(startTimeObserved);
			
		row->setEndTimeObserved(endTimeObserved);
			
		row->setCalDataType(calDataType);
	
		return row;		
	}	

	CalDataRow* CalDataTable::newRowFull(int numScan, vector<int > scanSet, CalTypeMod::CalType calType, ArrayTime startTimeObserved, ArrayTime endTimeObserved, CalDataOriginMod::CalDataOrigin calDataType)	{
		CalDataRow *row = new CalDataRow(*this);
			
		row->setNumScan(numScan);
			
		row->setScanSet(scanSet);
			
		row->setCalType(calType);
			
		row->setStartTimeObserved(startTimeObserved);
			
		row->setEndTimeObserved(endTimeObserved);
			
		row->setCalDataType(calDataType);
	
		return row;				
	}
	


CalDataRow* CalDataTable::newRow(CalDataRow* row) {
	return new CalDataRow(*this, *row);
}

CalDataRow* CalDataTable::newRowCopy(CalDataRow* row) {
	return new CalDataRow(*this, *row);
}

	//
	// Append a row to its table.
	//

	
	 
	
	/** 
 	 * Look up the table for a row whose noautoincrementable attributes are matching their
 	 * homologues in *x.  If a row is found  this row else autoincrement  *x.calDataId, 
 	 * add x to its table and returns x.
 	 *  
 	 * @returns a pointer on a CalDataRow.
 	 * @param x. A pointer on the row to be added.
 	 */ 
 		
			
	CalDataRow* CalDataTable::add(CalDataRow* x) {
			 
		CalDataRow* aRow = lookup(
				
		x->getNumScan()
				,
		x->getScanSet()
				,
		x->getCalType()
				,
		x->getStartTimeObserved()
				,
		x->getEndTimeObserved()
				,
		x->getCalDataType()
				
		);
		if (aRow) return aRow;
			

			
		// Autoincrement calDataId
		x->setCalDataId(Tag(size(), TagType::CalData));
						
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded();
		return x;
	}
		
		





	// 
	// A private method to append a row to its table, used by input conversion
	// methods.
	//

	
	/**
	 * If this table has an autoincrementable attribute then check if *x verifies the rule of uniqueness and throw exception if not.
	 * Check if *x verifies the key uniqueness rule and throw an exception if not.
	 * Append x to its table.
	 * @param x a pointer on the row to be appended.
	 * @returns a pointer on x.
	 */
	CalDataRow*  CalDataTable::checkAndAdd(CalDataRow* x) throw (DuplicateKey, UniquenessViolationException) {
	 
		 
		if (lookup(
			
			x->getNumScan()
		,
			x->getScanSet()
		,
			x->getCalType()
		,
			x->getStartTimeObserved()
		,
			x->getEndTimeObserved()
		,
			x->getCalDataType()
		
		)) throw UniquenessViolationException("Uniqueness violation exception in table CalDataTable");
		
		
		
		if (getRowByKey(
	
			x->getCalDataId()
			
		)) throw DuplicateKey("Duplicate key exception in ", "CalDataTable");
		
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded();
		return x;	
	}	







	

	//
	// ====> Methods returning rows.
	//	
	/**
	 * Get all rows.
	 * @return Alls rows as an array of CalDataRow
	 */
	vector<CalDataRow *> CalDataTable::get() {
		return privateRows;
		// return row;
	}

	
/*
 ** Returns a CalDataRow* given a key.
 ** @return a pointer to the row having the key whose values are passed as parameters, or 0 if
 ** no row exists for that key.
 **
 */
 	CalDataRow* CalDataTable::getRowByKey(Tag calDataId)  {
	CalDataRow* aRow = 0;
	for (unsigned int i = 0; i < row.size(); i++) {
		aRow = row.at(i);
		
			
				if (aRow->calDataId != calDataId) continue;
			
		
		return aRow;
	}
	return 0;		
}
	

	
/**
 * Look up the table for a row whose all attributes  except the autoincrementable one 
 * are equal to the corresponding parameters of the method.
 * @return a pointer on this row if any, 0 otherwise.
 *
			
 * @param numScan.
 	 		
 * @param scanSet.
 	 		
 * @param calType.
 	 		
 * @param startTimeObserved.
 	 		
 * @param endTimeObserved.
 	 		
 * @param calDataType.
 	 		 
 */
CalDataRow* CalDataTable::lookup(int numScan, vector<int > scanSet, CalTypeMod::CalType calType, ArrayTime startTimeObserved, ArrayTime endTimeObserved, CalDataOriginMod::CalDataOrigin calDataType) {
		CalDataRow* aRow;
		for (unsigned int i = 0; i < size(); i++) {
			aRow = row.at(i); 
			if (aRow->compareNoAutoInc(numScan, scanSet, calType, startTimeObserved, endTimeObserved, calDataType)) return aRow;
		}			
		return 0;	
} 
	
 	 	

	





#ifndef WITHOUT_ACS
	// Conversion Methods

	CalDataTableIDL *CalDataTable::toIDL() {
		CalDataTableIDL *x = new CalDataTableIDL ();
		unsigned int nrow = size();
		x->row.length(nrow);
		vector<CalDataRow*> v = get();
		for (unsigned int i = 0; i < nrow; ++i) {
			x->row[i] = *(v[i]->toIDL());
		}
		return x;
	}
#endif
	
#ifndef WITHOUT_ACS
	void CalDataTable::fromIDL(CalDataTableIDL x) throw(DuplicateKey,ConversionException) {
		unsigned int nrow = x.row.length();
		for (unsigned int i = 0; i < nrow; ++i) {
			CalDataRow *tmp = newRow();
			tmp->setFromIDL(x.row[i]);
			// checkAndAdd(tmp);
			add(tmp);
		}
	}
#endif

	char *CalDataTable::toFITS() const throw(ConversionException) {
		throw ConversionException("Not implemented","CalData");
	}

	void CalDataTable::fromFITS(char *fits) throw(ConversionException) {
		throw ConversionException("Not implemented","CalData");
	}

	string CalDataTable::toVOTable() const throw(ConversionException) {
		throw ConversionException("Not implemented","CalData");
	}

	void CalDataTable::fromVOTable(string vo) throw(ConversionException) {
		throw ConversionException("Not implemented","CalData");
	}

	string CalDataTable::toXML()  throw(ConversionException) {
		string buf;
		buf.append("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?> ");
//		buf.append("<CalDataTable xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"../../idl/CalDataTable.xsd\"> ");
		buf.append("<?xml-stylesheet type=\"text/xsl\" href=\"../asdm2html/table2html.xsl\"?> ");		
		buf.append("<CalDataTable> ");
		buf.append(entity.toXML());
		string s = container.getEntity().toXML();
		// Change the "Entity" tag to "ContainerEntity".
		buf.append("<Container" + s.substr(1,s.length() - 1)+" ");
		vector<CalDataRow*> v = get();
		for (unsigned int i = 0; i < v.size(); ++i) {
			try {
				buf.append(v[i]->toXML());
			} catch (NoSuchRow e) {
			}
			buf.append("  ");
		}		
		buf.append("</CalDataTable> ");
		return buf;
	}
	
	void CalDataTable::fromXML(string xmlDoc) throw(ConversionException) {
		Parser xml(xmlDoc);
		if (!xml.isStr("<CalDataTable")) 
			error();
		// cout << "Parsing a CalDataTable" << endl;
		string s = xml.getElement("<Entity","/>");
		if (s.length() == 0) 
			error();
		Entity e;
		e.setFromXML(s);
		if (e.getEntityTypeName() != "CalDataTable")
			error();
		setEntity(e);
		// Skip the container's entity; but, it has to be there.
		s = xml.getElement("<ContainerEntity","/>");
		if (s.length() == 0) 
			error();

		// Get each row in the table.
		s = xml.getElementContent("<row>","</row>");
		CalDataRow *row;
		while (s.length() != 0) {
			// cout << "Parsing a CalDataRow" << endl; 
			row = newRow();
			row->setFromXML(s);
			try {
				checkAndAdd(row);
			} catch (DuplicateKey e1) {
				throw ConversionException(e1.getMessage(),"CalDataTable");
			} 
			catch (UniquenessViolationException e1) {
				throw ConversionException(e1.getMessage(),"CalDataTable");	
			}
			catch (...) {
				// cout << "Unexpected error in CalDataTable::checkAndAdd called from CalDataTable::fromXML " << endl;
			}
			s = xml.getElementContent("<row>","</row>");
		}
		if (!xml.isStr("</CalDataTable>")) 
			error();
	}

	void CalDataTable::error() throw(ConversionException) {
		throw ConversionException("Invalid xml document","CalData");
	}
	
	string CalDataTable::toMIME() {
	 // To be implemented
		return "";
	}
	
	void CalDataTable::setFromMIME(const string & mimeMsg) {
		// To be implemented
		;
	}
	
	
	void CalDataTable::toFile(string directory) {
		if (!directoryExists(directory.c_str()) &&
			!createPath(directory.c_str())) {
			throw ConversionException("Could not create directory " , directory);
		}
		
		if (fileAsBin) {
			// write the bin serialized
			string fileName = directory + "/CalData.bin";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "CalData");
			tableout << toMIME() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "CalData");
		}
		else {
			// write the XML
			string fileName = directory + "/CalData.xml";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "CalData");
			tableout << toXML() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "CalData");
		}
	}
	
	void CalDataTable::setFromFile(const string& directory) {
		string tablename;
		if (fileAsBin)
			tablename = directory + "/CalData.bin";
		else
			tablename = directory + "/CalData.xml";
			
		// Determine the file size.
		ifstream::pos_type size;
		ifstream tablefile(tablename.c_str(), ios::in|ios::binary|ios::ate);

 		if (tablefile.is_open()) { 
  				size = tablefile.tellg(); 
  		}
		else {
				throw ConversionException("Could not open file " + tablename, "CalData");
		}
		
		// Re position to the beginning.
		tablefile.seekg(0);
		
		// Read in a stringstream.
		stringstream ss;
		ss << tablefile.rdbuf();

		if (tablefile.rdstate() == istream::failbit || tablefile.rdstate() == istream::badbit) {
			throw ConversionException("Error reading file " + tablename,"CalData");
		}

		// And close
		tablefile.close();
		if (tablefile.rdstate() == istream::failbit)
			throw ConversionException("Could not close file " + tablename,"CalData");
					
		// And parse the content with the appropriate method
		if (fileAsBin) 
			setFromMIME(ss.str());
		else
			fromXML(ss.str());	
	}			
			
	
	

	
	void CalDataTable::autoIncrement(string key, CalDataRow* x) {
		map<string, int>::iterator iter;
		if ((iter=noAutoIncIds.find(key)) == noAutoIncIds.end()) {
			// There is not yet a combination of the non autoinc attributes values in the hashtable
			
			// Initialize  calDataId to Tag(0).
			x->setCalDataId(Tag(0,  TagType::CalData));
			
			// Record it in the map.		
			noAutoIncIds.insert(make_pair(key, 0));			
		} 
		else {
			// There is already a combination of the non autoinc attributes values in the hashtable
			// Increment its value.
			int n = iter->second + 1; 
			
			// Initialize  calDataId to Tag(n).
			x->setCalDataId(Tag(n, TagType::CalData));
			
			// Record it in the map.		
			noAutoIncIds.insert(make_pair(key, n));				
		}		
	}
	
} // End namespace asdm
 
