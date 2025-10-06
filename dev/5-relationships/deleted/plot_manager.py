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

    def build_visual_relationships_bubble_2(self, agg_df):
        """
        Create a bubble plot showing average words vs average punctuations per message,
        with bubble size as number of messages, split by has_emoji. Top 5 group-author
        combinations by message_count are colored in gradients (green for has_emoji=True,
        red for has_emoji=False), others in gray. Includes trend lines with standard
        deviation bands, with overlapping area in orange.
        """
        try:
            if agg_df is None or agg_df.empty:
                logger.error("No data provided for bubble plot v2.")
                return None
            emoji_true = agg_df[agg_df['has_emoji'] == True]
            emoji_false = agg_df[agg_df['has_emoji'] == False]
            top_5_true = emoji_true.nlargest(5, 'message_count')
            top_5_false = emoji_false.nlargest(5, 'message_count')
            top_5_true_keys = top_5_true.apply(lambda row: f"{row['whatsapp_group']}-{row['author']}", axis=1).tolist()
            top_5_false_keys = top_5_false.apply(lambda row: f"{row['whatsapp_group']}-{row['author']}", axis=1).tolist()
            fig, ax = plt.subplots(figsize=(12, 8))
            for _, row in agg_df.iterrows():
                group_author = f"{row['whatsapp_group']}-{row['author']}"
                has_emoji = row['has_emoji']
                size = np.sqrt(row['message_count']) * 60  # Scale bubble size by factor 3
                initials = ''.join([name[0] for name in row['author'].split()])
                if has_emoji and group_author in top_5_true_keys:
                    rank = top_5_true_keys.index(group_author)
                    color = 'green'
                elif not has_emoji and group_author in top_5_false_keys:
                    rank = top_5_false_keys.index(group_author)
                    color = 'red'
                else:
                    color = 'gray'
                ax.scatter(row['avg_words'], row['avg_punct'], s=size, color=color, alpha=0.5)
                if group_author in top_5_true_keys or group_author in top_5_false_keys:
                    ax.text(
                        row['avg_words'], row['avg_punct'], initials,
                        fontsize=8, ha='center', va='center',
                        color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                    )
            if len(agg_df) > 1:
                line_x = np.linspace(agg_df['avg_words'].min(), agg_df['avg_words'].max(), 100)
                # Initialize arrays for overlap calculation
                y_false_lower = np.zeros_like(line_x)
                y_false_upper = np.zeros_like(line_x)
                y_true_lower = np.zeros_like(line_x)
                y_true_upper = np.zeros_like(line_x)
                # Trendline and band for has_emoji=False (red)
                if len(emoji_false) > 1:
                    x_false = emoji_false['avg_words']
                    y_false = emoji_false['avg_punct']
                    slope_false, intercept_false = np.polyfit(x_false, y_false, 1)
                    line_y_false = slope_false * line_x + intercept_false
                    ax.plot(line_x, line_y_false, color='red', linestyle='--')
                    predicted_y_false = slope_false * x_false + intercept_false
                    residuals_false = y_false - predicted_y_false
                    std_false = np.std(residuals_false)
                    y_false_lower = line_y_false - std_false
                    y_false_upper = line_y_false + std_false
                    ax.fill_between(line_x, y_false_lower, y_false_upper, color='red', alpha=0.4)
                # Trendline and band for has_emoji=True (green)
                if len(emoji_true) > 1:
                    x_true = emoji_true['avg_words']
                    y_true = emoji_true['avg_punct']
                    slope_true, intercept_true = np.polyfit(x_true, y_true, 1)
                    line_y_true = slope_true * line_x + intercept_true
                    ax.plot(line_x, line_y_true, color='green', linestyle='--')
                    predicted_y_true = slope_true * x_true + intercept_true
                    residuals_true = y_true - predicted_y_true
                    std_true = np.std(residuals_true)
                    y_true_lower = line_y_true - std_true
                    y_true_upper = line_y_true + std_true
                    ax.fill_between(line_x, y_true_lower, y_true_upper, color='green', alpha=0.4)
                # Fill overlapping area in orange
                if len(emoji_false) > 1 and len(emoji_true) > 1:
                    # Find overlapping region: max of lower bounds, min of upper bounds
                    y_overlap_lower = np.maximum(y_false_lower, y_true_lower)
                    y_overlap_upper = np.minimum(y_false_upper, y_true_upper)
                    # Only fill where upper bound > lower bound (valid overlap)
                    valid_overlap = y_overlap_upper > y_overlap_lower
                    if valid_overlap.any():
                        ax.fill_between(line_x, y_overlap_lower, y_overlap_upper, where=valid_overlap, color='green', alpha=0.4)
            ax.set_xlabel('Average Number of Words per Message')
            ax.set_ylabel('Average Number of Punctuations per Message')
            ax.set_title("More words = More Punctuations, but Emojis reduce number of Punctuations!", fontsize=20, x=0.6, ha='center')
            from matplotlib.patches import Patch
            legend_elements = []
            total_false_msgs = int(emoji_false['message_count'].sum()) if not emoji_false.empty else 0
            top_5_false_msgs = int(top_5_false['message_count'].sum()) if not top_5_false.empty else 0
            total_true_msgs = int(emoji_true['message_count'].sum()) if not emoji_true.empty else 0
            top_5_true_msgs = int(top_5_true['message_count'].sum()) if not top_5_true.empty else 0
            legend_elements.append(plt.scatter([], [], s=100, color='red', label=f'Without emojis\nTotal: {total_false_msgs:,} msgs / Top 5: {top_5_false_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=100, color='green', label=f'With emojis\nTotal: {total_true_msgs:,} msgs / Top 5: {top_5_true_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=0, label='\nSize of bubble reflects number of msgs\n', alpha=0))
            legend_elements.append(plt.plot([], [], color='red', linestyle='--', label='Trend line without emojis')[0])
            legend_elements.append(plt.plot([], [], color='green', linestyle='--', label='Trend line with emojis')[0])
            legend_elements.append(Patch(facecolor='red', alpha=0.4, label='Std dev without emojis'))
            legend_elements.append(Patch(facecolor='green', alpha=0.4, label='Std dev with emojis'))
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            logger.info("Created bubble plot v2 successfully.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build bubble plot v2: {e}")
            return None

    def build_visual_relationships_bubble_3(self, agg_df):
        """
        Create 2 plots in 1.
        """
        if agg_df is None or agg_df.empty:
            logger.error("No data provided for bubble plot 2.")
            return None

        try:
            # Define color shades (consistent with original)
            green_shades = ['#00CC00', '#1AFF1A', '#33FF33', '#4DFF4D', '#66FF66', '#80FF80', '#99FF99', '#B3FFB3', '#CCFFCC', '#E6FFE6']
            red_shades = ['#CC0000', '#FF1A1A', '#FF3333', '#FF4D4D', '#FF6666', '#FF8080', '#FF9999', '#FFB3B3', '#FFCCCC', '#FFE6E6']

            # Create figure with two subplots side by side, sharing y-axis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
            
            # Left subplot: Original bubble plot
            emoji_true = agg_df[agg_df['has_emoji'] == True]
            emoji_false = agg_df[agg_df['has_emoji'] == False]
            
            top_5_true = emoji_true.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_true_keys = [f"{group}_{author}" for group, author in top_5_true]
            top_5_false = emoji_false.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_false_keys = [f"{group}_{author}" for group, author in top_5_false]
            
            for _, row in agg_df.iterrows():
                group = row['whatsapp_group']
                author = row['author']
                has_emoji = row['has_emoji']
                group_author = f"{group}_{author}"
                size = row['message_count'] * 10
                initials = f"{group[0].upper()}{''.join(word[0].upper() for word in author.split() if word)}"
                
                if has_emoji and group_author in top_5_true_keys:
                    rank = top_5_true_keys.index(group_author)
                    color = green_shades[rank]
                elif not has_emoji and group_author in top_5_false_keys:
                    rank = top_5_false_keys.index(group_author)
                    color = red_shades[rank]
                else:
                    color = green_shades[6] if has_emoji else red_shades[6]
                
                ax1.scatter(
                    row['avg_words'],
                    row['avg_punct'],
                    s=size,
                    color=color,
                    alpha=0.5
                )
                
                ax1.text(
                    row['avg_words'],
                    row['avg_punct'],
                    initials,
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )
            
            slope_false, intercept_false, slope_true, intercept_true = None, None, None, None
            line_x = np.linspace(agg_df['avg_words'].min(), agg_df['avg_words'].max(), 100)
            
            if len(agg_df) > 1:
                if len(emoji_false) > 1:
                    x_false = emoji_false['avg_words']
                    y_false = emoji_false['avg_punct']
                    slope_false, intercept_false = np.polyfit(x_false, y_false, 1)
                    line_y_false = slope_false * line_x + intercept_false
                    ax1.plot(line_x, line_y_false, color='red', linestyle='--')
                    ax2.plot(line_x, line_y_false, color='red', linestyle='--', label='Trend without emojis')
                
                if len(emoji_true) > 1:
                    x_true = emoji_true['avg_words']
                    y_true = emoji_true['avg_punct']
                    slope_true, intercept_true = np.polyfit(x_true, y_true, 1)
                    line_y_true = slope_true * line_x + intercept_true
                    ax1.plot(line_x, line_y_true, color='green', linestyle='--')
                    ax2.plot(line_x, line_y_true, color='green', linestyle='--', label='Trend with emojis')
            
            ax1.set_xlabel('Average Number of Words per Message')
            ax1.set_ylabel('Average Number of Punctuations per Message')
            ax1.set_title("Bubble Plot: Words vs Punctuations")

            # Right subplot: Only trend lines with additional lines
            ax2.set_xlabel('Average Number of Words per Message')
            ax2.set_title("Trend Lines with Reference at 1 Punctuation")
            ax2.legend()
            
            # Add horizontal dashed black line at y=1.0
            ax2.axhline(y=1.0, color='black', linestyle='--', label='y=1.0')
            
            # Calculate and add vertical lines where y=1.0 intersects the trend lines (if within x-range)
            if slope_false is not None and slope_false != 0:
                x_red = (1.0 - intercept_false) / slope_false
                if agg_df['avg_words'].min() <= x_red <= agg_df['avg_words'].max():
                    ax2.axvline(x=x_red, color='black', linestyle='--', label=f'Intersect red at x={x_red:.2f}')
            
            if slope_true is not None and slope_true != 0:
                x_green = (1.0 - intercept_true) / slope_true
                if agg_df['avg_words'].min() <= x_green <= agg_df['avg_words'].max():
                    ax2.axvline(x=x_green, color='black', linestyle='--', label=f'Intersect green at x={x_green:.2f}')
            
            # Set matching axis limits
            x_min, x_max = agg_df['avg_words'].min(), agg_df['avg_words'].max()
            ax1.set_xlim(x_min - 1, x_max + 1)
            ax2.set_xlim(x_min - 1, x_max + 1)
            
            y_min, y_max = agg_df['avg_punct'].min(), agg_df['avg_punct'].max()
            ax1.set_ylim(y_min - 0.5, y_max + 0.5)
            ax2.set_ylim(y_min - 0.5, y_max + 0.5)
            
            plt.tight_layout()
            plt.show()
            
            return fig
        except Exception as e:
            logger.exception(f"Failed to build bubble plot 2: {e}")
            return None

    def build_visual_correlation_heatmap(self, df, groups=None):
        """
        Create a heatmap showing the correlation between word count and punctuation count
        for messages with and without emojis.

        Args:
            df (pandas.DataFrame): DataFrame with columns: message_cleaned, has_emoji.
            groups (list, optional): List of group names to filter. Defaults to all groups.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the heatmap, or None if creation fails.
        """
        try:
            if df is None or df.empty:
                logger.error("No data provided for correlation heatmap.")
                return None

            # Filter by groups if provided
            if groups is not None:
                df = df[df['whatsapp_group'].isin(groups)].copy()
                if df.empty:
                    logger.error(f"No data found for groups {groups} in correlation heatmap.")
                    return None

            # Calculate word and punctuation counts
            def count_words(message):
                if not isinstance(message, str):
                    return 0
                # Add space before emoji sequences if not preceded by space
                emoji_pattern = ''.join(re.escape(char) for char in emoji.EMOJI_DATA.keys())
                message = re.sub(r'([^\s])([' + emoji_pattern + r'])', r'\1 \2', message)
                # Replace sequences of emojis with a single 'EMOJI'
                message = re.sub(r'[' + emoji_pattern + r']+', 'EMOJI', message)
                # Handle currency with decimals as one word (e.g., €5.50 or $5.50)
                message = re.sub(r'([€$]\s*\d+[.,]\d+)', lambda m: m.group(0).replace(' ', ''), message)
                # Split on spaces
                words = re.split(r'\s+', message.strip())
                return len([w for w in words if w])

            def count_punctuations(message):
                if not isinstance(message, str):
                    return 0
                # Replace repeated punctuation with a single instance
                message = re.sub(r'([!?.,;:])\1+', r'\1', message)
                # Count ! ? . , ; : but exclude decimal points in numbers
                punctuations = re.findall(r'(?<![\d])[!?.,;:](?![\d])', message)
                return len(punctuations)

            df['word_count'] = df['message_cleaned'].apply(count_words)
            df['punct_count'] = df['message_cleaned'].apply(count_punctuations)

            # Split data by has_emoji
            emoji_true = df[df['has_emoji'] == True][['word_count', 'punct_count']]
            emoji_false = df[df['has_emoji'] == False][['word_count', 'punct_count']]

            # Calculate correlation matrices
            corr_true = emoji_true.corr() if not emoji_true.empty else pd.DataFrame(np.nan, index=['word_count', 'punct_count'], columns=['word_count', 'punct_count'])
            corr_false = emoji_false.corr() if not emoji_false.empty else pd.DataFrame(np.nan, index=['word_count', 'punct_count'], columns=['word_count', 'punct_count'])

            # Combine into a single DataFrame for heatmap
            corr_data = pd.DataFrame({
                'With Emojis': corr_true.loc['word_count', 'punct_count'],
                'Without Emojis': corr_false.loc['word_count', 'punct_count']
            }, index=['Correlation'])

            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
            ax.set_title('Correlation: Words vs Punctuation (With vs Without Emojis)')
            ax.set_ylabel('')
            plt.tight_layout()
            plt.show()

            logger.info("Created correlation heatmap successfully.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build correlation heatmap: {e}")
            return None              