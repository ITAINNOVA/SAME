from SPARQLWrapper import SPARQLWrapper, JSON


class QueryLOD:

    """ Extracts structured and non-structured DBpedia information for film domain."""

    PREFIX_DBPEDIA_ONTOLOGY = 'PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>'
    PREFIX_DBPEDIA_PROPERTY = 'PREFIX dbpedia-owl: <http://dbpedia.org/property/>'
    PREFIX_DBPEDIA_TERMS = 'PREFIX dbpedia-owl: <http://purl.org/dc/terms/>'
    PREFIX_DBPEDIA_RDFS = 'PREFIX dbpedia-owl: <http://www.w3.org/2000/01/rdf-schema#>'
    PREFIX_DPR = 'PREFIX dbpr: <http://dbpedia.org/resource/>'

    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    def __get_text(self, results, feature):
        result_list = list()
        for result in results['results']['bindings']:
            result_list.append(result[feature]['value'])
        return result_list

    def get_abstract(self, resource, language='en'):

        """
        Gets a film abstract.
        :param resource: DBpedia URI of the film.
        :param language: languague of abstract.
        :return: the abstract of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?abstract 
                    WHERE {"""
        query += '<' + resource + '> '
        query += """dbpedia-owl:abstract ?abstract .
                    FILTER langMatches(lang(?abstract),'""" + language + """')}"""

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='abstract')

    def get_budget(self, resource):

        """
        Gets a film budget.
        :param resource: DBpedia URI of the film.
        :return: the budget of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?budget 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:budget ?budget}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='budget')

    def get_cinematography(self, resource):

        """
        Gets a film cinematography.
        :param resource: DBpedia URI of the film.
        :return: the cinematography of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?cinematography 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:cinematography ?cinematography}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='cinematography')

    def get_director(self, resource):

        """
        Gets the film director/s.
        :param resource: DBpedia URI of the film.
        :return: the director/s of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?director 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:director ?director}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='director')

    def get_distributor(self, resource):

        """
        Gets the film distributors/s.
        :param resource: DBpedia URI of the film.
        :return: the distributors/s of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?distributor 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:distributor ?distributor}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='distributor')

    def get_editing(self, resource):

        """
        Gets the editing for a film.
        :param resource: DBpedia URI of the film.
        :return: the editing the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?editing 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:editing ?editing}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='editing')

    def get_gross(self, resource):

        """
        Gets the film gross.
        :param resource: DBpedia URI of the film.
        :return: the gross of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?gross 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:gross ?gross}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='gross')

    def get_music_composer(self, resource):

        """
        Gets the music composer of the film.
        :param resource: DBpedia URI of the film.
        :return: the music composer of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?musicComposer 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:musicComposer ?musicComposer}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='musicComposer')

    def get_producer(self, resource):

        """
        Gets the producer of the film.
        :param resource: DBpedia URI of the film.
        :return: the producer of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?producer 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:producer ?producer}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='producer')

    def get_runtime(self, resource):

        """
        Gets the runtime of the film.
        :param resource: DBpedia URI of the film.
        :return: the runtime of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?runtime 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:runtime ?runtime}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='runtime')

    def get_starring(self, resource):

        """
        Gets the starring of the film.
        :param resource: DBpedia URI of the film.
        :return: the starring of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?starring 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:starring ?starring}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='starring')

    def get_wiki_page_id(self, resource):

        """
        Gets the wiki page id of the film.
        :param resource: DBpedia URI of the film.
        :return: the wiki page id of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?wikiPageID 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:wikiPageID ?wikiPageID}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='wikiPageID')

    def get_wiki_page_revision_id(self, resource):

        """
        Gets the wiki page revision id of the film.
        :param resource: DBpedia URI of the film.
        :return: the wiki page revision id of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?wikiPageRevisionID 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:wikiPageRevisionID ?wikiPageRevisionID}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='wikiPageRevisionID')

    def get_writer(self, resource):

        """
        Gets the writer of the film.
        :param resource: DBpedia URI of the film.
        :return: the writer of the specific film.
        """

        query = self.PREFIX_DBPEDIA_ONTOLOGY
        query += """SELECT DISTINCT ?writer 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:writer ?writer}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='writer')

    def get_caption(self, resource):

        """
        Gets the caption of the film.
        :param resource: DBpedia URI of the film.
        :return: the caption of the specific film.
        """

        query = self.PREFIX_DBPEDIA_PROPERTY
        query += """SELECT DISTINCT ?caption 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:caption ?caption}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='caption')

    def get_country(self, resource):

        """
        Gets the country from the film.
        :param resource: DBpedia URI of the film.
        :return: the country from the specific film.
        """

        query = self.PREFIX_DBPEDIA_PROPERTY
        query += """SELECT DISTINCT ?country 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:country ?country}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='country')

    def get_language(self, resource):

        """
        Gets the language of the film.
        :param resource: DBpedia URI of the film.
        :return: the language of the specific film.
        """

        query = self.PREFIX_DBPEDIA_PROPERTY
        query += """SELECT DISTINCT ?language 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:language ?language}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='language')

    def get_studio(self, resource):

        """
        Gets the film studio.
        :param resource: DBpedia URI of the film.
        :return: the film studio.
        """

        query = self.PREFIX_DBPEDIA_PROPERTY
        query += """SELECT DISTINCT ?studio 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:studio ?studio}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()

        return self.__get_text(results, feature='studio')

    def get_type(self, resource):

        """
        Gets the film type.
        :param resource: DBpedia URI of the film.
        :return: the film type.
        """

        query = self.PREFIX_DBPEDIA_PROPERTY
        query += """SELECT DISTINCT ?type 
                    WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:type ?type}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()

        return self.__get_text(results, feature='type')

    def get_subject(self, resource):

        """
        Gets the subject of the film.
        :param resource: DBpedia URI of the film.
        :return: the subject of the specific film.
        """


        query = self.PREFIX_DBPEDIA_TERMS
        query += """SELECT DISTINCT ?subject
                            WHERE {"""
        query += '<' + resource + '> '
        query += 'dbpedia-owl:subject ?subject}'

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='subject')

    def get_comment(self, resource, language='en'):

        """
        Gets the comment of the film.
        :param resource: DBpedia URI of the film.
        :return: the comment of the specific film.
        """

        query = self.PREFIX_DBPEDIA_RDFS
        query += """SELECT DISTINCT ?comment 
                    WHERE {"""
        query += '<' + resource + '> '
        query += """dbpedia-owl:comment ?comment .
                    FILTER langMatches(lang(?comment),'""" + language + """')}"""

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        return self.__get_text(results, feature='comment')
