
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
 * File CalDataRow.h
 */
 
#ifndef CalDataRow_CLASS
#define CalDataRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::CalDataRowIDL;
#endif

#include <Angle.h>
#include <AngularRate.h>
#include <ArrayTime.h>
#include <ArrayTimeInterval.h>
#include <Complex.h>
#include <Entity.h>
#include <EntityId.h>
#include <EntityRef.h>
#include <Flux.h>
#include <Frequency.h>
#include <Humidity.h>
#include <Interval.h>
#include <Length.h>
#include <Pressure.h>
#include <Speed.h>
#include <Tag.h>
#include <Temperature.h>
#include <ConversionException.h>
#include <NoSuchRow.h>
#include <IllegalAccessException.h>

/*
#include <Enumerations.h>
using namespace enumerations;
 */




	

	

	

	

	
#include "CCalType.h"
using namespace CalTypeMod;
	

	

	

	

	

	

	

	

	
#include "CScanIntent.h"
using namespace ScanIntentMod;
	

	

	
#include "CAssociatedCalNature.h"
using namespace AssociatedCalNatureMod;
	

	
#include "CCalDataOrigin.h"
using namespace CalDataOriginMod;
	



using asdm::Angle;
using asdm::AngularRate;
using asdm::ArrayTime;
using asdm::Complex;
using asdm::Entity;
using asdm::EntityId;
using asdm::EntityRef;
using asdm::Flux;
using asdm::Frequency;
using asdm::Humidity;
using asdm::Interval;
using asdm::Length;
using asdm::Pressure;
using asdm::Speed;
using asdm::Tag;
using asdm::Temperature;
using asdm::ConversionException;
using asdm::NoSuchRow;
using asdm::IllegalAccessException;

/*\file CalData.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::CalDataTable;

	

/**
 * The CalDataRow class is a row of a CalDataTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class CalDataRow {
friend class asdm::CalDataTable;

public:

	virtual ~CalDataRow();

	/**
	 * Return the table to which this row belongs.
	 */
	CalDataTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a CalDataRowIDL struct.
	 */
	CalDataRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct CalDataRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (CalDataRowIDL x) throw(ConversionException);
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string toXML() const;

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void setFromXML (string rowDoc) throw(ConversionException);
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute calDataId
	
	
	

	
 	/**
 	 * Get calDataId.
 	 * @return calDataId as Tag
 	 */
 	Tag getCalDataId() const;
	
 
 	
 	
	
	


	
	// ===> Attribute numScan
	
	
	

	
 	/**
 	 * Get numScan.
 	 * @return numScan as int
 	 */
 	int getNumScan() const;
	
 
 	
 	
 	/**
 	 * Set numScan with the specified int.
 	 * @param numScan The int value to which numScan is to be set.
 	 
 		
 			
 	 */
 	void setNumScan (int numScan);
  		
	
	
	


	
	// ===> Attribute frequencyGroup, which is optional
	
	
	
	/**
	 * The attribute frequencyGroup is optional. Return true if this attribute exists.
	 * @return true if and only if the frequencyGroup attribute exists. 
	 */
	bool isFrequencyGroupExists() const;
	

	
 	/**
 	 * Get frequencyGroup, which is optional.
 	 * @return frequencyGroup as int
 	 * @throws IllegalAccessException If frequencyGroup does not exist.
 	 */
 	int getFrequencyGroup() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set frequencyGroup with the specified int.
 	 * @param frequencyGroup The int value to which frequencyGroup is to be set.
 	 
 		
 	 */
 	void setFrequencyGroup (int frequencyGroup);
		
	
	
	
	/**
	 * Mark frequencyGroup, which is an optional field, as non-existent.
	 */
	void clearFrequencyGroup ();
	


	
	// ===> Attribute scanSet
	
	
	

	
 	/**
 	 * Get scanSet.
 	 * @return scanSet as vector<int >
 	 */
 	vector<int > getScanSet() const;
	
 
 	
 	
 	/**
 	 * Set scanSet with the specified vector<int >.
 	 * @param scanSet The vector<int > value to which scanSet is to be set.
 	 
 		
 			
 	 */
 	void setScanSet (vector<int > scanSet);
  		
	
	
	


	
	// ===> Attribute calType
	
	
	

	
 	/**
 	 * Get calType.
 	 * @return calType as CalTypeMod::CalType
 	 */
 	CalTypeMod::CalType getCalType() const;
	
 
 	
 	
 	/**
 	 * Set calType with the specified CalTypeMod::CalType.
 	 * @param calType The CalTypeMod::CalType value to which calType is to be set.
 	 
 		
 			
 	 */
 	void setCalType (CalTypeMod::CalType calType);
  		
	
	
	


	
	// ===> Attribute freqGroupName, which is optional
	
	
	
	/**
	 * The attribute freqGroupName is optional. Return true if this attribute exists.
	 * @return true if and only if the freqGroupName attribute exists. 
	 */
	bool isFreqGroupNameExists() const;
	

	
 	/**
 	 * Get freqGroupName, which is optional.
 	 * @return freqGroupName as string
 	 * @throws IllegalAccessException If freqGroupName does not exist.
 	 */
 	string getFreqGroupName() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set freqGroupName with the specified string.
 	 * @param freqGroupName The string value to which freqGroupName is to be set.
 	 
 		
 	 */
 	void setFreqGroupName (string freqGroupName);
		
	
	
	
	/**
	 * Mark freqGroupName, which is an optional field, as non-existent.
	 */
	void clearFreqGroupName ();
	


	
	// ===> Attribute fieldName, which is optional
	
	
	
	/**
	 * The attribute fieldName is optional. Return true if this attribute exists.
	 * @return true if and only if the fieldName attribute exists. 
	 */
	bool isFieldNameExists() const;
	

	
 	/**
 	 * Get fieldName, which is optional.
 	 * @return fieldName as string
 	 * @throws IllegalAccessException If fieldName does not exist.
 	 */
 	string getFieldName() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set fieldName with the specified string.
 	 * @param fieldName The string value to which fieldName is to be set.
 	 
 		
 	 */
 	void setFieldName (string fieldName);
		
	
	
	
	/**
	 * Mark fieldName, which is an optional field, as non-existent.
	 */
	void clearFieldName ();
	


	
	// ===> Attribute fieldCode, which is optional
	
	
	
	/**
	 * The attribute fieldCode is optional. Return true if this attribute exists.
	 * @return true if and only if the fieldCode attribute exists. 
	 */
	bool isFieldCodeExists() const;
	

	
 	/**
 	 * Get fieldCode, which is optional.
 	 * @return fieldCode as vector<string >
 	 * @throws IllegalAccessException If fieldCode does not exist.
 	 */
 	vector<string > getFieldCode() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set fieldCode with the specified vector<string >.
 	 * @param fieldCode The vector<string > value to which fieldCode is to be set.
 	 
 		
 	 */
 	void setFieldCode (vector<string > fieldCode);
		
	
	
	
	/**
	 * Mark fieldCode, which is an optional field, as non-existent.
	 */
	void clearFieldCode ();
	


	
	// ===> Attribute startTimeObserved
	
	
	

	
 	/**
 	 * Get startTimeObserved.
 	 * @return startTimeObserved as ArrayTime
 	 */
 	ArrayTime getStartTimeObserved() const;
	
 
 	
 	
 	/**
 	 * Set startTimeObserved with the specified ArrayTime.
 	 * @param startTimeObserved The ArrayTime value to which startTimeObserved is to be set.
 	 
 		
 			
 	 */
 	void setStartTimeObserved (ArrayTime startTimeObserved);
  		
	
	
	


	
	// ===> Attribute endTimeObserved
	
	
	

	
 	/**
 	 * Get endTimeObserved.
 	 * @return endTimeObserved as ArrayTime
 	 */
 	ArrayTime getEndTimeObserved() const;
	
 
 	
 	
 	/**
 	 * Set endTimeObserved with the specified ArrayTime.
 	 * @param endTimeObserved The ArrayTime value to which endTimeObserved is to be set.
 	 
 		
 			
 	 */
 	void setEndTimeObserved (ArrayTime endTimeObserved);
  		
	
	
	


	
	// ===> Attribute sourceName, which is optional
	
	
	
	/**
	 * The attribute sourceName is optional. Return true if this attribute exists.
	 * @return true if and only if the sourceName attribute exists. 
	 */
	bool isSourceNameExists() const;
	

	
 	/**
 	 * Get sourceName, which is optional.
 	 * @return sourceName as vector<string >
 	 * @throws IllegalAccessException If sourceName does not exist.
 	 */
 	vector<string > getSourceName() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set sourceName with the specified vector<string >.
 	 * @param sourceName The vector<string > value to which sourceName is to be set.
 	 
 		
 	 */
 	void setSourceName (vector<string > sourceName);
		
	
	
	
	/**
	 * Mark sourceName, which is an optional field, as non-existent.
	 */
	void clearSourceName ();
	


	
	// ===> Attribute sourceCode, which is optional
	
	
	
	/**
	 * The attribute sourceCode is optional. Return true if this attribute exists.
	 * @return true if and only if the sourceCode attribute exists. 
	 */
	bool isSourceCodeExists() const;
	

	
 	/**
 	 * Get sourceCode, which is optional.
 	 * @return sourceCode as vector<string >
 	 * @throws IllegalAccessException If sourceCode does not exist.
 	 */
 	vector<string > getSourceCode() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set sourceCode with the specified vector<string >.
 	 * @param sourceCode The vector<string > value to which sourceCode is to be set.
 	 
 		
 	 */
 	void setSourceCode (vector<string > sourceCode);
		
	
	
	
	/**
	 * Mark sourceCode, which is an optional field, as non-existent.
	 */
	void clearSourceCode ();
	


	
	// ===> Attribute scanIntent, which is optional
	
	
	
	/**
	 * The attribute scanIntent is optional. Return true if this attribute exists.
	 * @return true if and only if the scanIntent attribute exists. 
	 */
	bool isScanIntentExists() const;
	

	
 	/**
 	 * Get scanIntent, which is optional.
 	 * @return scanIntent as vector<ScanIntentMod::ScanIntent >
 	 * @throws IllegalAccessException If scanIntent does not exist.
 	 */
 	vector<ScanIntentMod::ScanIntent > getScanIntent() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set scanIntent with the specified vector<ScanIntentMod::ScanIntent >.
 	 * @param scanIntent The vector<ScanIntentMod::ScanIntent > value to which scanIntent is to be set.
 	 
 		
 	 */
 	void setScanIntent (vector<ScanIntentMod::ScanIntent > scanIntent);
		
	
	
	
	/**
	 * Mark scanIntent, which is an optional field, as non-existent.
	 */
	void clearScanIntent ();
	


	
	// ===> Attribute assocCalDataId, which is optional
	
	
	
	/**
	 * The attribute assocCalDataId is optional. Return true if this attribute exists.
	 * @return true if and only if the assocCalDataId attribute exists. 
	 */
	bool isAssocCalDataIdExists() const;
	

	
 	/**
 	 * Get assocCalDataId, which is optional.
 	 * @return assocCalDataId as Tag
 	 * @throws IllegalAccessException If assocCalDataId does not exist.
 	 */
 	Tag getAssocCalDataId() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set assocCalDataId with the specified Tag.
 	 * @param assocCalDataId The Tag value to which assocCalDataId is to be set.
 	 
 		
 	 */
 	void setAssocCalDataId (Tag assocCalDataId);
		
	
	
	
	/**
	 * Mark assocCalDataId, which is an optional field, as non-existent.
	 */
	void clearAssocCalDataId ();
	


	
	// ===> Attribute assocCalNature, which is optional
	
	
	
	/**
	 * The attribute assocCalNature is optional. Return true if this attribute exists.
	 * @return true if and only if the assocCalNature attribute exists. 
	 */
	bool isAssocCalNatureExists() const;
	

	
 	/**
 	 * Get assocCalNature, which is optional.
 	 * @return assocCalNature as AssociatedCalNatureMod::AssociatedCalNature
 	 * @throws IllegalAccessException If assocCalNature does not exist.
 	 */
 	AssociatedCalNatureMod::AssociatedCalNature getAssocCalNature() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set assocCalNature with the specified AssociatedCalNatureMod::AssociatedCalNature.
 	 * @param assocCalNature The AssociatedCalNatureMod::AssociatedCalNature value to which assocCalNature is to be set.
 	 
 		
 	 */
 	void setAssocCalNature (AssociatedCalNatureMod::AssociatedCalNature assocCalNature);
		
	
	
	
	/**
	 * Mark assocCalNature, which is an optional field, as non-existent.
	 */
	void clearAssocCalNature ();
	


	
	// ===> Attribute calDataType
	
	
	

	
 	/**
 	 * Get calDataType.
 	 * @return calDataType as CalDataOriginMod::CalDataOrigin
 	 */
 	CalDataOriginMod::CalDataOrigin getCalDataType() const;
	
 
 	
 	
 	/**
 	 * Set calDataType with the specified CalDataOriginMod::CalDataOrigin.
 	 * @param calDataType The CalDataOriginMod::CalDataOrigin value to which calDataType is to be set.
 	 
 		
 			
 	 */
 	void setCalDataType (CalDataOriginMod::CalDataOrigin calDataType);
  		
	
	
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	
	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this CalDataRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(int numScan, vector<int > scanSet, CalTypeMod::CalType calType, ArrayTime startTimeObserved, ArrayTime endTimeObserved, CalDataOriginMod::CalDataOrigin calDataType);
	
	

	
	bool compareRequiredValue(int numScan, vector<int > scanSet, CalTypeMod::CalType calType, ArrayTime startTimeObserved, ArrayTime endTimeObserved, CalDataOriginMod::CalDataOrigin calDataType); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the CalDataRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(CalDataRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	CalDataTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a CalDataRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	CalDataRow (CalDataTable &table);

	/**
	 * Create a CalDataRow using a copy constructor mechanism.
	 * <p>
	 * Given a CalDataRow row and a CalDataTable table, the method creates a new
	 * CalDataRow owned by table. Each attribute of the created row is a copy (deep)
	 * of the corresponding attribute of row. The method does not add the created
	 * row to its table, its simply parents it to table, a call to the add method
	 * has to be done in order to get the row added (very likely after having modified
	 * some of its attributes).
	 * If row is null then the method returns a row with default values for its attributes. 
	 *
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 * @param row  The row which is to be copied.
	 */
	 CalDataRow (CalDataTable &table, CalDataRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute calDataId
	
	

	Tag calDataId;

	
	
 	
 	/**
 	 * Set calDataId with the specified Tag value.
 	 * @param calDataId The Tag value to which calDataId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setCalDataId (Tag calDataId);
  		
	

	
	// ===> Attribute numScan
	
	

	int numScan;

	
	
 	

	
	// ===> Attribute frequencyGroup, which is optional
	
	
	bool frequencyGroupExists;
	

	int frequencyGroup;

	
	
 	

	
	// ===> Attribute scanSet
	
	

	vector<int > scanSet;

	
	
 	

	
	// ===> Attribute calType
	
	

	CalTypeMod::CalType calType;

	
	
 	

	
	// ===> Attribute freqGroupName, which is optional
	
	
	bool freqGroupNameExists;
	

	string freqGroupName;

	
	
 	

	
	// ===> Attribute fieldName, which is optional
	
	
	bool fieldNameExists;
	

	string fieldName;

	
	
 	

	
	// ===> Attribute fieldCode, which is optional
	
	
	bool fieldCodeExists;
	

	vector<string > fieldCode;

	
	
 	

	
	// ===> Attribute startTimeObserved
	
	

	ArrayTime startTimeObserved;

	
	
 	

	
	// ===> Attribute endTimeObserved
	
	

	ArrayTime endTimeObserved;

	
	
 	

	
	// ===> Attribute sourceName, which is optional
	
	
	bool sourceNameExists;
	

	vector<string > sourceName;

	
	
 	

	
	// ===> Attribute sourceCode, which is optional
	
	
	bool sourceCodeExists;
	

	vector<string > sourceCode;

	
	
 	

	
	// ===> Attribute scanIntent, which is optional
	
	
	bool scanIntentExists;
	

	vector<ScanIntentMod::ScanIntent > scanIntent;

	
	
 	

	
	// ===> Attribute assocCalDataId, which is optional
	
	
	bool assocCalDataIdExists;
	

	Tag assocCalDataId;

	
	
 	

	
	// ===> Attribute assocCalNature, which is optional
	
	
	bool assocCalNatureExists;
	

	AssociatedCalNatureMod::AssociatedCalNature assocCalNature;

	
	
 	

	
	// ===> Attribute calDataType
	
	

	CalDataOriginMod::CalDataOrigin calDataType;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	///////////
	// Links //
	///////////
	

};

} // End namespace asdm

#endif /* CalData_CLASS */
