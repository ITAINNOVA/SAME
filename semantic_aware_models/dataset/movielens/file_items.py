from semantic_aware_models.dataset.lod.query_lod import QueryLOD


class Items:

    """ Class for generating a mapping between MovieLens and dbpedia """

    def __init__(self, mapping_file_path, separator):
        self.query_lod = QueryLOD()
        self.separator = separator
        self.map_movielens2dbpedia = self.__get_map(mapping_file_path)

    def __get_map(self, mapping_file_path):

        """
        Generates a map with the mapping between MovieLens (title of ítems) and dbpedia (resource)
        :param mapping_file_path: path of mapping file.
        :return: the map with the mapping
        """
        map = dict()
        for line in open(mapping_file_path, "r"):
            array = line.split(self.separator)
            title = array[1]
            resource_dbpedia = array[2].replace('\n', '')
            map.update({title: resource_dbpedia})
        return map

    @staticmethod
    def verify_is_empty(variable_array):

        """
        Check if the value of the variable (array) is unknown (empty).
        :param variable_array: vector to check.
        """
        resource_part = 'http://dbpedia.org/resource/'
        final_variable = 'N/A'

        if len(variable_array) == 1:
            variable = variable_array[0]
            if '\n*' in variable:
                final_variable = variable.replace('\n* ', '|')
                final_variable = final_variable.replace('* ', '')
            elif resource_part in variable:
                final_variable = variable.replace(resource_part, '')
            else:
                final_variable = variable
        elif len(variable_array) > 1:
            final_variable = variable_array[0].replace(resource_part, '')
            for variable in variable_array[1:]:
                final_variable += '|' + variable.replace(resource_part, '')
        return final_variable


class UnstructuredItems(Items):

    """ Class for generating unstructured information files."""

    def __init__(self, mapping_file_path, separator):
        Items.__init__(self, mapping_file_path, separator)

    def generate_new_unstructured_features(self, input_file_path, output_file_path):

        """
        Generates a new item file, called movies_unstructured.dat, which contains only unstructured information.
        :param input_file_path: path of the input file.
        :param output_file_path: path of the output file.
        """
        result = ''
        resource_dbpedia = ''
        movies_unstructured_file = open(output_file_path, 'w', encoding='utf8')
        for line in open(input_file_path, "r"):
            array = line.split(self.separator)
            item_id = array[0]
            title = array[1]

            if title in self.map_movielens2dbpedia:
                resource_dbpedia = self.map_movielens2dbpedia[title]
                print(item_id + '::' + resource_dbpedia)
                abstract = Items.verify_is_empty(variable_array=self.query_lod.get_abstract(resource=resource_dbpedia))

                # Pre-processing
                abstract = abstract.replace('::', ' ')
                abstract = abstract.replace('#', ' ')
                abstract = abstract.replace(' \n*  ', ', ')

                if abstract != 'N/A':
                    result = item_id + self.separator + title + self.separator + abstract + '\n'
                    movies_unstructured_file.write(result)
                # else:
                #    result = item_id + self.separator + title + self.separator + 'N/A' + '\n'
        movies_unstructured_file.close()


class StructuredItems(Items):

    """ Class for generating structured information files."""

    def __init__(self, mapping_file_path, separator):
        Items.__init__(self, mapping_file_path, separator)

    def generate_new_structured_features(self, input_file_path, output_file_path):

        """
        Generates a new item file, called movies_structured.dat, which contains only structured information (or features)
        :param input_file_path: path of the input file.
        :param output_file_path: path of the output file.
        """

        result = ''
        resource_dbpedia = ''
        movies_structured_file = open(output_file_path, 'w', encoding='utf8')
        for line in open(input_file_path, "r"):
            line = line.replace('\r','').replace('\n', '')
            array = line.split(self.separator)
            item_id = array[0]
            title = array[1]
            genres = array[2]

            if title in self.map_movielens2dbpedia:
                resource_dbpedia = self.map_movielens2dbpedia[title]
                print(item_id + '::' + resource_dbpedia)

                budget = Items.verify_is_empty(variable_array=self.query_lod.get_budget(resource=resource_dbpedia))
                cinematography = Items.verify_is_empty(variable_array=self.query_lod.get_cinematography(resource=resource_dbpedia))
                director = Items.verify_is_empty(variable_array=self.query_lod.get_director(resource=resource_dbpedia))
                distributor = Items.verify_is_empty(variable_array=self.query_lod.get_distributor(resource=resource_dbpedia))
                editing = Items.verify_is_empty(variable_array=self.query_lod.get_editing(resource=resource_dbpedia))
                gross = Items.verify_is_empty(variable_array=self.query_lod.get_gross(resource=resource_dbpedia))
                music_composer = Items.verify_is_empty(variable_array=self.query_lod.get_music_composer(resource=resource_dbpedia))
                producer = Items.verify_is_empty(variable_array=self.query_lod.get_producer(resource=resource_dbpedia))
                runtime = Items.verify_is_empty(variable_array=self.query_lod.get_runtime(resource=resource_dbpedia))
                starring = Items.verify_is_empty(variable_array=self.query_lod.get_starring(resource=resource_dbpedia))
                wiki_page_id = Items.verify_is_empty(variable_array=self.query_lod.get_wiki_page_id(resource=resource_dbpedia))
                wiki_page_revision_id = Items.verify_is_empty(variable_array=self.query_lod.get_wiki_page_revision_id(resource=resource_dbpedia))
                writer = Items.verify_is_empty(variable_array=self.query_lod.get_writer(resource=resource_dbpedia))
                caption = Items.verify_is_empty(variable_array=self.query_lod.get_caption(resource=resource_dbpedia))
                country = Items.verify_is_empty(variable_array=self.query_lod.get_country(resource=resource_dbpedia))
                language = Items.verify_is_empty(variable_array=self.query_lod.get_language(resource=resource_dbpedia))
                studio = Items.verify_is_empty(variable_array=self.query_lod.get_studio(resource=resource_dbpedia))
                type_film = Items.verify_is_empty(variable_array=self.query_lod.get_type(resource=resource_dbpedia))
                # comment = Items.verify_is_empty(variable_array=self.query_lod.get_comment(resource=resource_dbpedia))
                subject = Items.verify_is_empty(variable_array=self.query_lod.get_subject(resource=resource_dbpedia))

                result = item_id + self.separator + title + self.separator + genres + self.separator + \
                         budget + self.separator + \
                         cinematography + self.separator + \
                         director + self.separator + \
                         distributor + self.separator + \
                         editing + self.separator + \
                         gross + self.separator + \
                         music_composer + self.separator + \
                         producer + self.separator + \
                         runtime + self.separator + \
                         starring + self.separator + \
                         wiki_page_id + self.separator + \
                         wiki_page_revision_id + self.separator + \
                         writer + self.separator + \
                         caption + self.separator + \
                         country + self.separator + \
                         language + self.separator + \
                         studio + self.separator + \
                         type_film + self.separator + \
                         subject + '\n'
            else:
                result = item_id + self.separator + title + self.separator + genres + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + self.separator + \
                         'N/A' + '\n'
            movies_structured_file.write(result)
        movies_structured_file.close()
