    def build_visual_categories_2(self, df):
        """
        Create a bar chart showing the number of messages by attachment type, including a 'No attachments' category.

        Args:
            df (pandas.DataFrame): The DataFrame containing message data with relevant columns.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        try:
            # Compute counts for each attachment type
            counts = {}
            counts['Links'] = df['has_link'].sum()
            counts['Pictures'] = (df['pictures_deleted'] > 0).sum()
            counts['Videos'] = (df['videos_deleted'] > 0).sum()
            counts['Audios'] = (df['audios_deleted'] > 0).sum()
            counts['Gifs'] = (df['gifs_deleted'] > 0).sum()
            counts['Stickers'] = (df['stickers_deleted'] > 0).sum()
            counts['Documents'] = (df['documents_deleted'] > 0).sum()
            counts['Videonotes'] = (df['videonotes_deleted'] > 0).sum()

            # Compute 'No attachments': messages where none of the above conditions are true
            no_attach_mask = ~df['has_link'] & \
                            (df['pictures_deleted'] == 0) & \
                            (df['videos_deleted'] == 0) & \
                            (df['audios_deleted'] == 0) & \
                            (df['gifs_deleted'] == 0) & \
                            (df['stickers_deleted'] == 0) & \
                            (df['documents_deleted'] == 0) & \
                            (df['videonotes_deleted'] == 0)
            counts['No attachments'] = no_attach_mask.sum()

            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            categories = list(counts.keys())
            values = list(counts.values())
            ax.bar(categories, values, color='skyblue')
            ax.set_xlabel("Attachment Type")
            ax.set_ylabel("Number of Messages")
            ax.set_title("Messages by Attachment Type (July 2020 - July 2025)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build attachment categories bar chart: {e}")
            return None

    def build_visual_distribution_2(self, df):
        """
        Create a scatterplot showing the number of messages containing each of the top 25 emojis
        for each author in the 'maap' WhatsApp group. Each author is represented by a different color.

        Args:
            df (pandas.DataFrame): DataFrame filtered for 'maap' group, containing 'author' and 'message' columns.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the scatterplot, or None if creation fails.
        """
        if df.empty:
            logger.error("Empty DataFrame provided for build_visual_distribution_2.")
            return None

        try:
            # Recreate ignore_emojis (skin tone modifiers) from DataEditor
            ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}

            # Extract all emojis from messages, excluding ignored ones
            all_emojis = []
            for message in df['message']:
                if isinstance(message, str):
                    all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA and c not in ignore_emojis])

            if not all_emojis:
                logger.warning("No emojis found in the messages for 'maap' group.")
                return None

            # Get top 25 emojis by occurrence frequency
            emoji_freq = Counter(all_emojis)
            top_25_emojis = [e for e, _ in emoji_freq.most_common(25)]

            # Get sorted unique authors (assuming 4 authors)
            authors = sorted(df['author'].unique())
            if len(authors) != 4:
                logger.warning(f"Expected 4 authors in 'maap' group, found {len(authors)}.")

            # Define colors for each author
            colors = ['blue', 'red', 'green', 'purple']  # Distinct colors for up to 4 authors
            color_map = {author: color for author, color in zip(authors, colors)}

            # Compute counts: number of messages containing each top emoji per author
            counts = {author: [] for author in authors}
            for emoji_char in top_25_emojis:
                for author in authors:
                    author_msgs = df[df['author'] == author]['message']
                    count = sum(1 for msg in author_msgs if isinstance(msg, str) and emoji_char in msg)
                    counts[author].append(count)

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            x_pos = np.arange(len(top_25_emojis))

            # Plot scatter points for each author
            max_y = 0
            for author in authors:
                y = counts[author]
                ax.scatter(x_pos, y, color=color_map[author], label=author, s=50)
                max_y = max(max_y, max(y) if y else 0)

            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(top_25_emojis, fontsize=14)
            ax.set_xlabel('Top 25 Emojis (Ordered by Overall Frequency)')
            ax.set_ylabel('Number of Messages')
            ax.set_title('Number of Messages Containing Top 25 Emojis per Author in "maap" Group (July 2020 - July 2025)')
            ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 1)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build visual distribution 2: {e}")
            return None

    def build_visual_relationships(self, table2, group):
        """
        Create a horizontal bar chart showing total counts and highest author counts for first words
        with high association to one author, including the author with the highest percentage.

        Args:
            table2 (pandas.DataFrame): Numerical DataFrame from build_visual_relationships (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted desc).
            group (str): WhatsApp group name for the title.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        if table2 is None or table2.empty:
            logger.error("No valid table provided for building visual relationships plot.")
            return None

        try:
            # Sort by highest descending, reverse for plotting (highest at top)
            table2 = table2.sort_values('highest', ascending=True)  # Ascending for barh (bottom to top)
            first_words = table2.index

            # Identify the author with the highest percentage for each first word
            author_columns = [col for col in table2.columns if col not in ['total', 'highest']]
            highest_authors = table2[author_columns].idxmax(axis=1)

            # Combine first word and highest author for y-axis labels
            y_labels = [f"{word} ({author})" for word, author in zip(first_words, highest_authors)]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, max(4, len(first_words) * 0.5)))

            # Black bars for total
            ax.barh(y_labels, table2['total'], color='black', label='Total Count')

            # Orange bars for highest count (highest % * total / 100)
            highest_counts = (table2['highest'] / 100) * table2['total']
            ax.barh(y_labels, highest_counts, color='orange', label='Highest Author Count')

            ax.set_xlabel('Count')
            ax.set_ylabel('First Word (Author with Highest %)')
            ax.set_title(f"First Words Strongly Associated with One Author in {group}")
            ax.legend()
            plt.tight_layout()

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build visual relationships plot: {e}")
            return None

    def build_visual_relationships_2(self, table2, group):
        """
        Create a horizontal bar chart showing total counts and highest author counts for emoji sequences
        with high association to one author, including the author with the highest percentage.

        Args:
            table2 (pandas.DataFrame): Numerical DataFrame from build_visual_relationships_2 (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted desc).
            group (str): WhatsApp group name for the title.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        if table2 is None or table2.empty:
            logger.error("No valid table provided for building visual relationships_2 plot.")
            return None

        try:
            # Sort by highest descending, reverse for plotting (highest at top)
            table2 = table2.sort_values('highest', ascending=True)  # Ascending for barh (bottom to top)
            sequences = table2.index

            # Identify the author with the highest percentage for each sequence
            author_columns = [col for col in table2.columns if col not in ['total', 'highest']]
            highest_authors = table2[author_columns].idxmax(axis=1)

            # Combine sequence and highest author for y-axis labels
            y_labels = [f"{seq} ({author})" for seq, author in zip(sequences, highest_authors)]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, max(4, len(sequences) * 0.5)))

            # Black bars for total
            ax.barh(y_labels, table2['total'], color='black', label='Total Count')

            # Orange bars for highest count (highest % * total / 100)
            highest_counts = (table2['highest'] / 100) * table2['total']
            ax.barh(y_labels, highest_counts, color='orange', label='Highest Author Count')

            ax.set_xlabel('Count')
            ax.set_ylabel('Emoji Sequence (Author with Highest %)')
            ax.set_title(f"Emoji Sequences Strongly Associated with One Author in {group}")
            ax.legend()
            plt.tight_layout()

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build visual relationships_2 plot: {e}")
            return None

    def build_visual_relationships_network(self, combined_df, group):
        """
        Create a network diagram showing interactions between authors in the 'maap' group.

        Args:
            combined_df (pandas.DataFrame): DataFrame from build_visual_relationships_3 with 'Pairs' rows containing interaction data.
            group (str): WhatsApp group name ('maap').

        Returns:
            matplotlib.figure.Figure or None: Figure object for the network diagram, or None if creation fails.
        """
        if nx is None:
            logger.error("Cannot create network diagram: NetworkX is not installed.")
            return None
        if combined_df is None or combined_df.empty:
            logger.error("No valid DataFrame provided for building visual relationships_3 plot.")
            return None
        try:
            # Extract Pairs data
            pairs_df = combined_df[combined_df['type'] == 'Pairs'][['author', 'total_messages']]
            if pairs_df.empty:
                logger.error("No Pairs data found in combined_df for building network diagram.")
                return None

            # Create graph
            G = nx.Graph()
            authors = set()
            for pair in pairs_df['author']:
                a1, a2 = pair.split(' & ')
                a1, a2 = a1.strip(), a2.strip()
                authors.add(a1)
                authors.add(a2)
                G.add_edge(a1, a2, weight=pairs_df[pairs_df['author'] == pair]['total_messages'].iloc[0])

            # Define married couples (assumption based on common naming conventions)
            married_edges = [('Anja Berkemeijer', 'Phons Berkemeijer'), ('Madeleine', 'Anthony van Tilburg')]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Draw edges (non-married in blue, married in red)
            max_weight = max(nx.get_edge_attributes(G, 'weight').values(), default=1)
            for u, v in G.edges():
                edge_data = G.get_edge_data(u, v)
                if edge_data is None:
                    logger.warning(f"Edge ({u}, {v}) not found in graph. Skipping.")
                    continue
                weight = edge_data['weight']
                color = 'red' if (u, v) in married_edges or (v, u) in married_edges else 'blue'
                width = 1 + 5 * (weight / max_weight)  # Scale width between 1 and 6
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, ax=ax)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            # Add edge labels
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            ax.set_title(f"Messaging Interactions in {group} Group\n(Red: Married Couples, Blue: Others)")
            plt.axis('off')
            plt.tight_layout()

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build network diagram: {e}")
            return None        