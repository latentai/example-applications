
#include <sodium.h>
#include <string>
#include <iostream>
#include <fstream>    
#include <vector>
#include <tvm/runtime/registry.h>

/*!
 * \brief Generates a hash from a given password
 * \param password Password for generation of hash
 * \param hash Reference to hash to be generated
 */
void generate_hash(std::string &password,std::vector<unsigned char> &hash);

/*!
 * \brief Unencrypts a data buffer
 * \param source_file Encrypted file
 * \param key_hash Key used for encryption
 * \param chunk_size size of data to be unencrypted
 * \param decrypted_buff unencrypted data
 * 
 * \return True if successful 
 */

bool lre_decrypt_buff(std::string &source_file, std::vector<unsigned char> &key_hash,const unsigned int &chunk_size, std::vector<unsigned char> &decrypted_buff);

/*!
 * \brief Unlocks the locked key to unencrypt the model generated from LEIP SDK
 * \param password Password used for encryption of key
 * \param key_file Path to locked key file
 */
std::vector<unsigned char> unlock_key(std::string &password, std::string &key_file);
