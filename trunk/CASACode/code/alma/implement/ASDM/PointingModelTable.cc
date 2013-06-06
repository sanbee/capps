
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
 * File PointingModelTable.cpp
 */
#include <ConversionException.h>
#include <DuplicateKey.h>
#include <OutOfBoundsException.h>

using asdm::ConversionException;
using asdm::DuplicateKey;
using asdm::OutOfBoundsException;

#include <ASDM.h>
#include <PointingModelTable.h>
#include <PointingModelRow.h>
#include <Parser.h>

using asdm::ASDM;
using asdm::PointingModelTable;
using asdm::PointingModelRow;
using asdm::Parser;

#include <iostream>
#include <sstream>
#include <set>
using namespace std;

#include <Misc.h>
using namespace asdm;

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "boost/filesystem/operations.hpp"


namespace asdm {

	string PointingModelTable::tableName = "PointingModel";
	const vector<string> PointingModelTable::attributesNames = initAttributesNames();
		

	/**
	 * The list of field names that make up key key.
	 * (Initialization is in the constructor.)
	 */
	vector<string> PointingModelTable::key;

	/**
	 * Return the list of field names that make up key key
	 * as an array of strings.
	 */	
	vector<string> PointingModelTable::getKeyName() {
		return key;
	}


	PointingModelTable::PointingModelTable(ASDM &c) : container(c) {

	
		key.push_back("antennaId");
	
		key.push_back("pointingModelId");
	


		// Define a default entity.
		entity.setEntityId(EntityId("uid://X0/X0/X0"));
		entity.setEntityIdEncrypted("na");
		entity.setEntityTypeName("PointingModelTable");
		entity.setEntityVersion("1");
		entity.setInstanceVersion("1");
		
		// Archive XML
		archiveAsBin = false;
		
		// File XML
		fileAsBin = false;
	}
	
/**
 * A destructor for PointingModelTable.
 */
	PointingModelTable::~PointingModelTable() {
		for (unsigned int i = 0; i < privateRows.size(); i++) 
			delete(privateRows.at(i));
	}

	/**
	 * Container to which this table belongs.
	 */
	ASDM &PointingModelTable::getContainer() const {
		return container;
	}

	/**
	 * Return the number of rows in the table.
	 */
	unsigned int PointingModelTable::size() {
		return privateRows.size();
	}
	
	/**
	 * Return the name of this table.
	 */
	string PointingModelTable::getName() const {
		return tableName;
	}
	
	/**
	 * Build the vector of attributes names.
	 */
	vector<string> PointingModelTable::initAttributesNames() {
		vector<string> attributesNames;

		attributesNames.push_back("antennaId");

		attributesNames.push_back("pointingModelId");


		attributesNames.push_back("numCoeff");

		attributesNames.push_back("coeffName");

		attributesNames.push_back("coeffVal");

		attributesNames.push_back("polarizationType");

		attributesNames.push_back("receiverBand");

		attributesNames.push_back("assocNature");

		attributesNames.push_back("assocPointingModelId");


		attributesNames.push_back("coeffFormula");

		return attributesNames;
	}
	
	/**
	 * Return the names of the attributes.
	 */
	const vector<string>& PointingModelTable::getAttributesNames() { return attributesNames; }

	/**
	 * Return this table's Entity.
	 */
	Entity PointingModelTable::getEntity() const {
		return entity;
	}

	/**
	 * Set this table's Entity.
	 */
	void PointingModelTable::setEntity(Entity e) {
		this->entity = e; 
	}
	
	//
	// ====> Row creation.
	//
	
	/**
	 * Create a new row.
	 */
	PointingModelRow *PointingModelTable::newRow() {
		return new PointingModelRow (*this);
	}
	

	/**
	 * Create a new row initialized to the specified values.
	 * @return a pointer on the created and initialized row.
	
 	 * @param antennaId 
	
 	 * @param numCoeff 
	
 	 * @param coeffName 
	
 	 * @param coeffVal 
	
 	 * @param polarizationType 
	
 	 * @param receiverBand 
	
 	 * @param assocNature 
	
 	 * @param assocPointingModelId 
	
     */
	PointingModelRow* PointingModelTable::newRow(Tag antennaId, int numCoeff, vector<string > coeffName, vector<float > coeffVal, PolarizationTypeMod::PolarizationType polarizationType, ReceiverBandMod::ReceiverBand receiverBand, string assocNature, int assocPointingModelId){
		PointingModelRow *row = new PointingModelRow(*this);
			
		row->setAntennaId(antennaId);
			
		row->setNumCoeff(numCoeff);
			
		row->setCoeffName(coeffName);
			
		row->setCoeffVal(coeffVal);
			
		row->setPolarizationType(polarizationType);
			
		row->setReceiverBand(receiverBand);
			
		row->setAssocNature(assocNature);
			
		row->setAssocPointingModelId(assocPointingModelId);
	
		return row;		
	}	
	


PointingModelRow* PointingModelTable::newRow(PointingModelRow* row) {
	return new PointingModelRow(*this, *row);
}

	//
	// Append a row to its table.
	//

	
	 
	
	/** 
 	 * Look up the table for a row whose noautoincrementable attributes are matching their
 	 * homologues in *x.  If a row is found  this row else autoincrement  *x.pointingModelId, 
 	 * add x to its table and returns x.
 	 *  
 	 * @returns a pointer on a PointingModelRow.
 	 * @param x. A pointer on the row to be added.
 	 */ 
 		
			
	PointingModelRow* PointingModelTable::add(PointingModelRow* x) {
		PointingModelRow* aRow = lookup(
				
			x->getAntennaId()
				,
			x->getNumCoeff()
				,
			x->getCoeffName()
				,
			x->getCoeffVal()
				,
			x->getPolarizationType()
				,
			x->getReceiverBand()
				,
			x->getAssocNature()
				,
			x->getAssocPointingModelId()
				
		);
		if (aRow) return aRow;

		// Autoincrementation algorithm. We use the hashtable.
		ostringstream noAutoIncIdsEntry;
		noAutoIncIdsEntry
			
				
		<< x->antennaId.toString() << "_"
					
			
		;
		
		map<string, int>::iterator iter;
		if ((iter=noAutoIncIds.find(noAutoIncIdsEntry.str())) == noAutoIncIds.end()) {
			// There is not yet a combination of the non autoinc attributes values in the hashtable

			
			// Initialize  pointingModelId to 0.
			x->pointingModelId = 0;		
			//x->setPointingModelId(0);
			
			// Record it in the map.		
			noAutoIncIds.insert(make_pair(noAutoIncIdsEntry.str(), 0));			
		} 
		else {
			// There is already a combination of the non autoinc attributes values in the hashtable
			// Increment its value.
			int n = iter->second + 1; 
			
			// Initialize  pointingModelId to n.		
			//x->setPointingModelId(n);
			x->pointingModelId = n;
			
			// Record it in the map.		
			noAutoIncIds.erase(iter);
			noAutoIncIds.insert(make_pair(noAutoIncIdsEntry.str(), n));				
		}	
		
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded(true);
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
	 * @throws DuplicateKey
	 
	 * @throws UniquenessViolationException
	 
	 */
	PointingModelRow*  PointingModelTable::checkAndAdd(PointingModelRow* x)  {
	 
		 
		if (lookup(
			
			x->getAntennaId()
		,
			x->getNumCoeff()
		,
			x->getCoeffName()
		,
			x->getCoeffVal()
		,
			x->getPolarizationType()
		,
			x->getReceiverBand()
		,
			x->getAssocNature()
		,
			x->getAssocPointingModelId()
		
		)) throw UniquenessViolationException("Uniqueness violation exception in table PointingModelTable");
		
		
		
		if (getRowByKey(
	
			x->getAntennaId()
	,
			x->getPointingModelId()
			
		)) throw DuplicateKey("Duplicate key exception in ", "PointingModelTable");
		
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded(true);
		return x;	
	}	







	

	//
	// ====> Methods returning rows.
	//	
	/**
	 * Get all rows.
	 * @return Alls rows as an array of PointingModelRow
	 */
	vector<PointingModelRow *> PointingModelTable::get() {
		return privateRows;
		// return row;
	}

	
/*
 ** Returns a PointingModelRow* given a key.
 ** @return a pointer to the row having the key whose values are passed as parameters, or 0 if
 ** no row exists for that key.
 **
 */
 	PointingModelRow* PointingModelTable::getRowByKey(Tag antennaId, int pointingModelId)  {
	PointingModelRow* aRow = 0;
	for (unsigned int i = 0; i < row.size(); i++) {
		aRow = row.at(i);
		
			
				if (aRow->antennaId != antennaId) continue;
			
		
			
				if (aRow->pointingModelId != pointingModelId) continue;
			
		
		return aRow;
	}
	return 0;		
}
	

	
/**
 * Look up the table for a row whose all attributes  except the autoincrementable one 
 * are equal to the corresponding parameters of the method.
 * @return a pointer on this row if any, 0 otherwise.
 *
			
 * @param antennaId.
 	 		
 * @param numCoeff.
 	 		
 * @param coeffName.
 	 		
 * @param coeffVal.
 	 		
 * @param polarizationType.
 	 		
 * @param receiverBand.
 	 		
 * @param assocNature.
 	 		
 * @param assocPointingModelId.
 	 		 
 */
PointingModelRow* PointingModelTable::lookup(Tag antennaId, int numCoeff, vector<string > coeffName, vector<float > coeffVal, PolarizationTypeMod::PolarizationType polarizationType, ReceiverBandMod::ReceiverBand receiverBand, string assocNature, int assocPointingModelId) {
		PointingModelRow* aRow;
		for (unsigned int i = 0; i < size(); i++) {
			aRow = row.at(i); 
			if (aRow->compareNoAutoInc(antennaId, numCoeff, coeffName, coeffVal, polarizationType, receiverBand, assocNature, assocPointingModelId)) return aRow;
		}			
		return 0;	
} 
	
 	 	

	 	
/*
 * Returns a vector of pointers on rows whose key element pointingModelId 
 * is equal to the parameter pointingModelId.
 * @return a vector of vector <PointingModelRow *>. A returned vector of size 0 means that no row has been found.
 * @param pointingModelId int contains the value of
 * the autoincrementable attribute that is looked up in the table.
 */
 vector <PointingModelRow *>  PointingModelTable::getRowByPointingModelId(int pointingModelId) {
	vector<PointingModelRow *> list;
	for (unsigned int i = 0; i < row.size(); ++i) {
		PointingModelRow &x = *row[i];
					
		if (x.pointingModelId == pointingModelId)
			
		list.push_back(row[i]);
	}
	//if (list.size() == 0) throw  NoSuchRow("","PointingModel");
	return list;	
 }
	




#ifndef WITHOUT_ACS
	// Conversion Methods

	PointingModelTableIDL *PointingModelTable::toIDL() {
		PointingModelTableIDL *x = new PointingModelTableIDL ();
		unsigned int nrow = size();
		x->row.length(nrow);
		vector<PointingModelRow*> v = get();
		for (unsigned int i = 0; i < nrow; ++i) {
			x->row[i] = *(v[i]->toIDL());
		}
		return x;
	}
#endif
	
#ifndef WITHOUT_ACS
	void PointingModelTable::fromIDL(PointingModelTableIDL x) {
		unsigned int nrow = x.row.length();
		for (unsigned int i = 0; i < nrow; ++i) {
			PointingModelRow *tmp = newRow();
			tmp->setFromIDL(x.row[i]);
			// checkAndAdd(tmp);
			add(tmp);
		}
	}
#endif

	
	string PointingModelTable::toXML()  {
		string buf;

		buf.append("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?> ");
		buf.append("<PointingModelTable xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:pointm=\"http://Alma/XASDM/PointingModelTable\" xsi:schemaLocation=\"http://Alma/XASDM/PointingModelTable http://almaobservatory.org/XML/XASDM/2/PointingModelTable.xsd\" schemaVersion=\"2\" schemaRevision=\"1.54\">\n");
	
		buf.append(entity.toXML());
		string s = container.getEntity().toXML();
		// Change the "Entity" tag to "ContainerEntity".
		buf.append("<Container" + s.substr(1,s.length() - 1)+" ");
		vector<PointingModelRow*> v = get();
		for (unsigned int i = 0; i < v.size(); ++i) {
			try {
				buf.append(v[i]->toXML());
			} catch (NoSuchRow e) {
			}
			buf.append("  ");
		}		
		buf.append("</PointingModelTable> ");
		return buf;
	}

	
	void PointingModelTable::fromXML(string xmlDoc)  {
		Parser xml(xmlDoc);
		if (!xml.isStr("<PointingModelTable")) 
			error();
		// cout << "Parsing a PointingModelTable" << endl;
		string s = xml.getElement("<Entity","/>");
		if (s.length() == 0) 
			error();
		Entity e;
		e.setFromXML(s);
		if (e.getEntityTypeName() != "PointingModelTable")
			error();
		setEntity(e);
		// Skip the container's entity; but, it has to be there.
		s = xml.getElement("<ContainerEntity","/>");
		if (s.length() == 0) 
			error();

		// Get each row in the table.
		s = xml.getElementContent("<row>","</row>");
		PointingModelRow *row;
		while (s.length() != 0) {
			// cout << "Parsing a PointingModelRow" << endl; 
			row = newRow();
			row->setFromXML(s);
			try {
				checkAndAdd(row);
			} catch (DuplicateKey e1) {
				throw ConversionException(e1.getMessage(),"PointingModelTable");
			} 
			catch (UniquenessViolationException e1) {
				throw ConversionException(e1.getMessage(),"PointingModelTable");	
			}
			catch (...) {
				// cout << "Unexpected error in PointingModelTable::checkAndAdd called from PointingModelTable::fromXML " << endl;
			}
			s = xml.getElementContent("<row>","</row>");
		}
		if (!xml.isStr("</PointingModelTable>")) 
			error();
			
		archiveAsBin = false;
		fileAsBin = false;
		
	}

	
	void PointingModelTable::error()  {
		throw ConversionException("Invalid xml document","PointingModel");
	}
	
	
	string PointingModelTable::MIMEXMLPart(const asdm::ByteOrder* byteOrder) {
		string UID = getEntity().getEntityId().toString();
		string withoutUID = UID.substr(6);
		string containerUID = getContainer().getEntity().getEntityId().toString();
		ostringstream oss;
		oss << "<?xml version='1.0'  encoding='ISO-8859-1'?>";
		oss << "\n";
		oss << "<PointingModelTable xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:pointm=\"http://Alma/XASDM/PointingModelTable\" xsi:schemaLocation=\"http://Alma/XASDM/PointingModelTable http://almaobservatory.org/XML/XASDM/2/PointingModelTable.xsd\" schemaVersion=\"2\" schemaRevision=\"1.54\">\n";
		oss<< "<Entity entityId='"<<UID<<"' entityIdEncrypted='na' entityTypeName='PointingModelTable' schemaVersion='1' documentVersion='1'/>\n";
		oss<< "<ContainerEntity entityId='"<<containerUID<<"' entityIdEncrypted='na' entityTypeName='ASDM' schemaVersion='1' documentVersion='1'/>\n";
		oss << "<BulkStoreRef file_id='"<<withoutUID<<"' byteOrder='"<<byteOrder->toString()<<"' />\n";
		oss << "<Attributes>\n";

		oss << "<antennaId/>\n"; 
		oss << "<pointingModelId/>\n"; 
		oss << "<numCoeff/>\n"; 
		oss << "<coeffName/>\n"; 
		oss << "<coeffVal/>\n"; 
		oss << "<polarizationType/>\n"; 
		oss << "<receiverBand/>\n"; 
		oss << "<assocNature/>\n"; 
		oss << "<assocPointingModelId/>\n"; 

		oss << "<coeffFormula/>\n"; 
		oss << "</Attributes>\n";		
		oss << "</PointingModelTable>\n";

		return oss.str();				
	}
	
	string PointingModelTable::toMIME(const asdm::ByteOrder* byteOrder) {
		EndianOSStream eoss(byteOrder);
		
		string UID = getEntity().getEntityId().toString();
		
		// The MIME Header
		eoss <<"MIME-Version: 1.0";
		eoss << "\n";
		eoss << "Content-Type: Multipart/Related; boundary='MIME_boundary'; type='text/xml'; start= '<header.xml>'";
		eoss <<"\n";
		eoss <<"Content-Description: Correlator";
		eoss <<"\n";
		eoss <<"alma-uid:" << UID;
		eoss <<"\n";
		eoss <<"\n";		
		
		// The MIME XML part header.
		eoss <<"--MIME_boundary";
		eoss <<"\n";
		eoss <<"Content-Type: text/xml; charset='ISO-8859-1'";
		eoss <<"\n";
		eoss <<"Content-Transfer-Encoding: 8bit";
		eoss <<"\n";
		eoss <<"Content-ID: <header.xml>";
		eoss <<"\n";
		eoss <<"\n";
		
		// The MIME XML part content.
		eoss << MIMEXMLPart(byteOrder);

		// The MIME binary part header
		eoss <<"--MIME_boundary";
		eoss <<"\n";
		eoss <<"Content-Type: binary/octet-stream";
		eoss <<"\n";
		eoss <<"Content-ID: <content.bin>";
		eoss <<"\n";
		eoss <<"\n";	
		
		// The MIME binary content
		entity.toBin(eoss);
		container.getEntity().toBin(eoss);
		eoss.writeInt((int) privateRows.size());
		for (unsigned int i = 0; i < privateRows.size(); i++) {
			privateRows.at(i)->toBin(eoss);	
		}
		
		// The closing MIME boundary
		eoss << "\n--MIME_boundary--";
		eoss << "\n";
		
		return eoss.str();	
	}

	
	void PointingModelTable::setFromMIME(const string & mimeMsg) {
    string xmlPartMIMEHeader = "Content-ID: <header.xml>\n\n";
    
    string binPartMIMEHeader = "--MIME_boundary\nContent-Type: binary/octet-stream\nContent-ID: <content.bin>\n\n";
    
    // Detect the XML header.
    string::size_type loc0 = mimeMsg.find(xmlPartMIMEHeader, 0);
    if ( loc0 == string::npos) {
      throw ConversionException("Failed to detect the beginning of the XML header", "PointingModel");
    }
    loc0 += xmlPartMIMEHeader.size();
    
    // Look for the string announcing the binary part.
    string::size_type loc1 = mimeMsg.find( binPartMIMEHeader, loc0 );
    
    if ( loc1 == string::npos ) {
      throw ConversionException("Failed to detect the beginning of the binary part", "PointingModel");
    }
    
    //
    // Extract the xmlHeader and analyze it to find out what is the byte order and the sequence
    // of attribute names.
    //
    string xmlHeader = mimeMsg.substr(loc0, loc1-loc0);
    xmlDoc *doc;
    doc = xmlReadMemory(xmlHeader.data(), xmlHeader.size(), "BinaryTableHeader.xml", NULL, XML_PARSE_NOBLANKS);
    if ( doc == NULL ) 
      throw ConversionException("Failed to parse the xmlHeader into a DOM structure.", "PointingModel");
    
   // This vector will be filled by the names of  all the attributes of the table
   // in the order in which they are expected to be found in the binary representation.
   //
    vector<string> attributesSeq;
      
    xmlNode* root_element = xmlDocGetRootElement(doc);
    if ( root_element == NULL || root_element->type != XML_ELEMENT_NODE )
      throw ConversionException("Failed to parse the xmlHeader into a DOM structure.", "PointingModel");
    
    const ByteOrder* byteOrder;
    if ( string("ASDMBinaryTable").compare((const char*) root_element->name) == 0) {
      // Then it's an "old fashioned" MIME file for tables.
      // Just try to deserialize it with Big_Endian for the bytes ordering.
      byteOrder = asdm::ByteOrder::Big_Endian;
      
 	 //
    // Let's consider a  default order for the sequence of attributes.
    //
     
    attributesSeq.push_back("antennaId") ; 
     
    attributesSeq.push_back("pointingModelId") ; 
     
    attributesSeq.push_back("numCoeff") ; 
     
    attributesSeq.push_back("coeffName") ; 
     
    attributesSeq.push_back("coeffVal") ; 
     
    attributesSeq.push_back("polarizationType") ; 
     
    attributesSeq.push_back("receiverBand") ; 
     
    attributesSeq.push_back("assocNature") ; 
     
    attributesSeq.push_back("assocPointingModelId") ; 
    
     
    attributesSeq.push_back("coeffFormula") ; 
              
     }
    else if (string("PointingModelTable").compare((const char*) root_element->name) == 0) {
      // It's a new (and correct) MIME file for tables.
      //
      // 1st )  Look for a BulkStoreRef element with an attribute byteOrder.
      //
      xmlNode* bulkStoreRef = 0;
      xmlNode* child = root_element->children;
      
      // Skip the two first children (Entity and ContainerEntity).
      bulkStoreRef = (child ==  0) ? 0 : ( (child->next) == 0 ? 0 : child->next->next );
      
      if ( bulkStoreRef == 0 || (bulkStoreRef->type != XML_ELEMENT_NODE)  || (string("BulkStoreRef").compare((const char*) bulkStoreRef->name) != 0))
      	throw ConversionException ("Could not find the element '/PointingModelTable/BulkStoreRef'. Invalid XML header '"+ xmlHeader + "'.", "PointingModel");
      	
      // We found BulkStoreRef, now look for its attribute byteOrder.
      _xmlAttr* byteOrderAttr = 0;
      for (struct _xmlAttr* attr = bulkStoreRef->properties; attr; attr = attr->next) 
	  if (string("byteOrder").compare((const char*) attr->name) == 0) {
	   byteOrderAttr = attr;
	   break;
	 }
      
      if (byteOrderAttr == 0) 
	     throw ConversionException("Could not find the element '/PointingModelTable/BulkStoreRef/@byteOrder'. Invalid XML header '" + xmlHeader +"'.", "PointingModel");
      
      string byteOrderValue = string((const char*) byteOrderAttr->children->content);
      if (!(byteOrder = asdm::ByteOrder::fromString(byteOrderValue)))
		throw ConversionException("No valid value retrieved for the element '/PointingModelTable/BulkStoreRef/@byteOrder'. Invalid XML header '" + xmlHeader + "'.", "PointingModel");
		
	 //
	 // 2nd) Look for the Attributes element and grab the names of the elements it contains.
	 //
	 xmlNode* attributes = bulkStoreRef->next;
     if ( attributes == 0 || (attributes->type != XML_ELEMENT_NODE)  || (string("Attributes").compare((const char*) attributes->name) != 0))	 
       	throw ConversionException ("Could not find the element '/PointingModelTable/Attributes'. Invalid XML header '"+ xmlHeader + "'.", "PointingModel");
 
 	xmlNode* childOfAttributes = attributes->children;
 	
 	while ( childOfAttributes != 0 && (childOfAttributes->type == XML_ELEMENT_NODE) ) {
 		attributesSeq.push_back(string((const char*) childOfAttributes->name));
 		childOfAttributes = childOfAttributes->next;
    }
    }
    // Create an EndianISStream from the substring containing the binary part.
    EndianISStream eiss(mimeMsg.substr(loc1+binPartMIMEHeader.size()), byteOrder);
    
    entity = Entity::fromBin(eiss);
    
    // We do nothing with that but we have to read it.
    Entity containerEntity = Entity::fromBin(eiss);
    
    int numRows = eiss.readInt();
    try {
      for (int i = 0; i < numRows; i++) {
	PointingModelRow* aRow = PointingModelRow::fromBin(eiss, *this, attributesSeq);
	checkAndAdd(aRow);
      }
    }
    catch (DuplicateKey e) {
      throw ConversionException("Error while writing binary data , the message was "
				+ e.getMessage(), "PointingModel");
    }
    catch (TagFormatException e) {
      throw ConversionException("Error while reading binary data , the message was "
				+ e.getMessage(), "PointingModel");
    }
    archiveAsBin = true;
    fileAsBin = true;
	}

	
	void PointingModelTable::toFile(string directory) {
		if (!directoryExists(directory.c_str()) &&
			!createPath(directory.c_str())) {
			throw ConversionException("Could not create directory " , directory);
		}

		string fileName = directory + "/PointingModel.xml";
		ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
		if (tableout.rdstate() == ostream::failbit)
			throw ConversionException("Could not open file " + fileName + " to write ", "PointingModel");
		if (fileAsBin) 
			tableout << MIMEXMLPart();
		else
			tableout << toXML() << endl;
		tableout.close();
		if (tableout.rdstate() == ostream::failbit)
			throw ConversionException("Could not close file " + fileName, "PointingModel");

		if (fileAsBin) {
			// write the bin serialized
			string fileName = directory + "/PointingModel.bin";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "PointingModel");
			tableout << toMIME() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "PointingModel");
		}
	}

	
	void PointingModelTable::setFromFile(const string& directory) {		
    if (boost::filesystem::exists(boost::filesystem::path(uniqSlashes(directory + "/PointingModel.xml"))))
      setFromXMLFile(directory);
    else if (boost::filesystem::exists(boost::filesystem::path(uniqSlashes(directory + "/PointingModel.bin"))))
      setFromMIMEFile(directory);
    else
      throw ConversionException("No file found for the PointingModel table", "PointingModel");
	}			

	
  void PointingModelTable::setFromMIMEFile(const string& directory) {
    string tablePath ;
    
    tablePath = directory + "/PointingModel.bin";
    ifstream tablefile(tablePath.c_str(), ios::in|ios::binary);
    if (!tablefile.is_open()) { 
      throw ConversionException("Could not open file " + tablePath, "PointingModel");
    }
    // Read in a stringstream.
    stringstream ss; ss << tablefile.rdbuf();
    
    if (tablefile.rdstate() == istream::failbit || tablefile.rdstate() == istream::badbit) {
      throw ConversionException("Error reading file " + tablePath,"PointingModel");
    }
    
    // And close.
    tablefile.close();
    if (tablefile.rdstate() == istream::failbit)
      throw ConversionException("Could not close file " + tablePath,"PointingModel");
    
    setFromMIME(ss.str());
  }	

	
void PointingModelTable::setFromXMLFile(const string& directory) {
    string tablePath ;
    
    tablePath = directory + "/PointingModel.xml";
    ifstream tablefile(tablePath.c_str(), ios::in|ios::binary);
    if (!tablefile.is_open()) { 
      throw ConversionException("Could not open file " + tablePath, "PointingModel");
    }
      // Read in a stringstream.
    stringstream ss;
    ss << tablefile.rdbuf();
    
    if  (tablefile.rdstate() == istream::failbit || tablefile.rdstate() == istream::badbit) {
      throw ConversionException("Error reading file '" + tablePath + "'", "PointingModel");
    }
    
    // And close
    tablefile.close();
    if (tablefile.rdstate() == istream::failbit)
      throw ConversionException("Could not close file '" + tablePath + "'", "PointingModel");

    // Let's make a string out of the stringstream content and empty the stringstream.
    string xmlDocument = ss.str(); ss.str("");

    // Let's make a very primitive check to decide
    // whether the XML content represents the table
    // or refers to it via a <BulkStoreRef element.
    if (xmlDocument.find("<BulkStoreRef") != string::npos)
      setFromMIMEFile(directory);
    else
      fromXML(xmlDocument);
  }

	

	

			
	
	

	
	void PointingModelTable::autoIncrement(string key, PointingModelRow* x) {
		map<string, int>::iterator iter;
		if ((iter=noAutoIncIds.find(key)) == noAutoIncIds.end()) {
			// There is not yet a combination of the non autoinc attributes values in the hashtable
			
			// Initialize  pointingModelId to 0.		
			x->setPointingModelId(0);
			
			// Record it in the map.		
			noAutoIncIds.insert(make_pair(key, 0));			
		} 
		else {
			// There is already a combination of the non autoinc attributes values in the hashtable
			// Increment its value.
			int n = iter->second + 1; 
			
			// Initialize  pointingModelId to n.		
			x->setPointingModelId(n);
			
			// Record it in the map.		
			noAutoIncIds.insert(make_pair(key, n));				
		}		
	}
	
} // End namespace asdm
 