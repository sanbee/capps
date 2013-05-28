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
 * File UniquenessViolationException.h
 */

#ifndef UniquenessViolationException_CLASS
#define UniquenessViolationException_CLASS

#include <string>
using std::string;

namespace asdm {
  /**
   * The UniquenessViolationException class represents an exception  
   * occuring when one tries to add a row in table whose all mandatory
   * attributes values except the autoincrementable one are already 
   * met in an existing row of the table. 
   */
  class UniquenessViolationException {
    
  public:
    
    /**
     * An empty constructor.
     */
    UniquenessViolationException();

    /**
     * Create an exception exception  
     * occuring when one tries to add a row in table whose all mandatory
     * attributes values except the autoincrementable one are already 
     * met in an existing row of the table.
     * 
     * @param t string. Name the table where the exception occurred.  
     */
    UniquenessViolationException(string t);

    
    virtual ~UniquenessViolationException();

    string getMessage() const;
    
  protected:
    
    string message;	
    
  };
  
  inline UniquenessViolationException::UniquenessViolationException(): message("Uniqueness violation exception") {;}
  inline UniquenessViolationException::UniquenessViolationException (string t) : 
    message("The uniqueness constraint on all attributes except the autoincrementable one is violated in table "+t) {
  }
  
  inline UniquenessViolationException::~UniquenessViolationException() {;}
  inline string UniquenessViolationException::getMessage() const {
    return message;
  }
  
} // End namespace asdm

#endif /* UniquenessViolationException_CLASS */